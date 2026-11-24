//! Session permission memory and `ControlHandler` bridging `can_use_tool` onto
//! `ClientPeer::request_permission`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::SessionId;
use aither_acp::domain::{ConfigValueId, PermissionOptionId};
use aither_acp::domain::{
    CurrentModeUpdate, Open, PermissionOption, PermissionOptionKind, PermissionOutcome,
    PermissionRequest, PermissionSubject, SessionUpdate, ToolCallStatus, ToolCallSubject,
};
use aither_acp::server::SessionContext;
use claude_acp_sdk::{
    ControlHandler, Permission, PermissionBehavior, PermissionDecision, PermissionDestination,
    PermissionModeName, PermissionType, ToolPermissionRequest,
};
use serde_json::Value;

use crate::convert::build_tool_call;
use crate::mode::AcpMode;

const OPT_ALLOW_ONCE: &str = "allow_once";
const OPT_ALLOW_ALWAYS: &str = "allow_always";
const OPT_REJECT_ONCE: &str = "reject_once";
const OPT_REJECT_ALWAYS: &str = "reject_always";

#[derive(Debug, Clone, PartialEq, Eq)]
enum RememberedVerdict {
    Allow,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RememberedRule {
    tool_name: String,
    pattern: Option<String>,
    verdict: RememberedVerdict,
}

impl RememberedRule {
    fn matches(&self, tool_name: &str, input: &Value) -> bool {
        if self.tool_name != tool_name {
            return false;
        }
        let Some(pattern) = &self.pattern else {
            return true;
        };
        if pattern == "*" || pattern == "**" {
            return true;
        }
        let Some(subject) = rule_subject(tool_name, input) else {
            return false;
        };
        wildcard_match(pattern, subject)
    }
}

/// Per-session store of `AllowAlways` and `RejectAlways` tool rules.
#[derive(Debug, Default)]
pub struct PermissionMemory {
    rules: Mutex<Vec<RememberedRule>>,
    accept_edits_granted: AtomicBool,
}

impl PermissionMemory {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Clear all remembered session rules (e.g. on `/clear` conversation reset).
    pub(crate) fn clear(&self) {
        self.rules
            .lock()
            .expect("permission memory mutex poisoned")
            .clear();
        self.accept_edits_granted.store(false, Ordering::SeqCst);
    }

    /// Check whether `req` is already decided by a prior `AllowAlways` or
    /// `RejectAlways` selection in this session.
    pub(crate) fn evaluate(&self, req: &ToolPermissionRequest) -> Option<PermissionDecision> {
        let rules = self.rules.lock().expect("permission memory mutex poisoned");
        // Deny rules take precedence over allow rules.
        if rules
            .iter()
            .any(|r| r.verdict == RememberedVerdict::Deny && r.matches(&req.tool_name, &req.input))
        {
            drop(rules);
            return Some(PermissionDecision::deny(format!(
                "tool `{}` was rejected for this session",
                req.tool_name
            )));
        }

        let allowed_by_rule = rules.iter().any(|r| {
            r.verdict == RememberedVerdict::Allow && r.matches(&req.tool_name, &req.input)
        });
        drop(rules);

        if allowed_by_rule {
            return Some(PermissionDecision::allow());
        }

        if self.accept_edits_granted.load(Ordering::SeqCst)
            && matches!(
                req.tool_name.as_str(),
                "Edit" | "MultiEdit" | "Write" | "NotebookEdit"
            )
        {
            return Some(PermissionDecision::allow());
        }

        None
    }

    /// Persist an `AllowAlways` grant for `req` and return any wire
    /// `updatedPermissions` derived from `req.permission_suggestions` plus
    /// any mode promotion (`SetMode`).
    pub(crate) fn record_allow_always(
        &self,
        req: &ToolPermissionRequest,
    ) -> (Vec<Permission>, Option<AcpMode>) {
        let mut updated_permissions = Vec::new();
        let mut promoted_mode = None;
        let mut recorded_any_rule = false;

        {
            let mut rules = self.rules.lock().expect("permission memory mutex poisoned");
            for suggestion in &req.permission_suggestions {
                let mut perm = Permission::from_suggestion(suggestion);
                // Scope editor `AllowAlways` grants to the session so we never
                // mutate user/project settings files behind the caller's back.
                perm.destination = PermissionDestination::Session;

                if perm.permission_type == PermissionType::SetMode {
                    match &perm.mode {
                        Some(PermissionModeName::AcceptEdits) => {
                            self.accept_edits_granted.store(true, Ordering::SeqCst);
                            promoted_mode = Some(AcpMode::WorkspaceWrite);
                            recorded_any_rule = true;
                        }
                        Some(PermissionModeName::BypassPermissions) => {
                            promoted_mode = Some(AcpMode::AgentFullAccess);
                            recorded_any_rule = true;
                        }
                        Some(PermissionModeName::Unknown(mode)) if mode == "auto" => {
                            promoted_mode = Some(AcpMode::Auto);
                            recorded_any_rule = true;
                        }
                        Some(PermissionModeName::Unknown(mode)) if mode == "dontAsk" => {
                            promoted_mode = Some(AcpMode::DontAsk);
                            recorded_any_rule = true;
                        }
                        _ => {}
                    }
                }

                if perm.permission_type == PermissionType::AddRules
                    && perm
                        .behavior
                        .as_ref()
                        .is_none_or(|b| *b == PermissionBehavior::Allow)
                    && let Some(perm_rules) = &perm.rules
                {
                    for r in perm_rules {
                        push_unique_rule(
                            &mut rules,
                            RememberedRule {
                                tool_name: r.tool_name.clone(),
                                pattern: (!r.rule_content.is_empty())
                                    .then(|| r.rule_content.clone()),
                                verdict: RememberedVerdict::Allow,
                            },
                        );
                        recorded_any_rule = true;
                    }
                }

                updated_permissions.push(perm);
            }

            if !recorded_any_rule {
                push_unique_rule(
                    &mut rules,
                    RememberedRule {
                        tool_name: req.tool_name.clone(),
                        pattern: None,
                        verdict: RememberedVerdict::Allow,
                    },
                );
            }
        }

        (updated_permissions, promoted_mode)
    }

    /// Persist a `RejectAlways` rule for `req` in this session.
    pub(crate) fn record_reject_always(&self, req: &ToolPermissionRequest) {
        let mut rules = self.rules.lock().expect("permission memory mutex poisoned");
        let mut recorded = false;
        for suggestion in &req.permission_suggestions {
            if suggestion.suggestion_type == PermissionType::AddRules
                && let Some(raw_rules) = &suggestion.rules
            {
                for item in raw_rules {
                    if let Some(tool) = item.get("toolName").and_then(Value::as_str) {
                        let pat = item
                            .get("ruleContent")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .map(ToOwned::to_owned);
                        push_unique_rule(
                            &mut rules,
                            RememberedRule {
                                tool_name: tool.to_string(),
                                pattern: pat,
                                verdict: RememberedVerdict::Deny,
                            },
                        );
                        recorded = true;
                    }
                }
            }
        }

        if !recorded {
            push_unique_rule(
                &mut rules,
                RememberedRule {
                    tool_name: req.tool_name.clone(),
                    pattern: None,
                    verdict: RememberedVerdict::Deny,
                },
            );
        }
    }
}

fn push_unique_rule(rules: &mut Vec<RememberedRule>, candidate: RememberedRule) {
    if !rules.contains(&candidate) {
        rules.push(candidate);
    }
}

fn rule_subject<'a>(tool_name: &str, input: &'a Value) -> Option<&'a str> {
    match tool_name {
        "Bash" => input.get("command").and_then(Value::as_str),
        "Read" | "Edit" | "MultiEdit" | "Write" => input.get("file_path").and_then(Value::as_str),
        "NotebookRead" | "NotebookEdit" => input.get("notebook_path").and_then(Value::as_str),
        "Glob" | "Grep" | "LS" => input
            .get("path")
            .or_else(|| input.get("pattern"))
            .and_then(Value::as_str),
        "WebFetch" => input.get("url").and_then(Value::as_str),
        "WebSearch" | "ToolSearch" => input.get("query").and_then(Value::as_str),
        _ => None,
    }
}

/// Simple wildcard matcher supporting `*` (any sequence of characters, including
/// `**` path globs) used for Claude CLI `ruleContent` patterns such as
/// `"cargo *"` or `"/workspace/**"`.
fn wildcard_match(pattern: &str, candidate: &str) -> bool {
    let pat_chars: Vec<char> = pattern.chars().collect();
    let cand_chars: Vec<char> = candidate.chars().collect();
    let (mut p_idx, mut c_idx) = (0_usize, 0_usize);
    let mut star_idx: Option<usize> = None;
    let mut match_idx = 0_usize;

    while c_idx < cand_chars.len() {
        if p_idx < pat_chars.len() && pat_chars[p_idx] == cand_chars[c_idx] {
            p_idx += 1;
            c_idx += 1;
        } else if p_idx < pat_chars.len() && pat_chars[p_idx] == '*' {
            while p_idx < pat_chars.len() && pat_chars[p_idx] == '*' {
                p_idx += 1;
            }
            star_idx = Some(p_idx);
            match_idx = c_idx;
        } else if let Some(after_star) = star_idx {
            p_idx = after_star;
            match_idx += 1;
            c_idx = match_idx;
        } else {
            return false;
        }
    }

    while p_idx < pat_chars.len() && pat_chars[p_idx] == '*' {
        p_idx += 1;
    }
    p_idx == pat_chars.len()
}

fn perm_option(id: &str, name: &str, kind: PermissionOptionKind) -> PermissionOption {
    PermissionOption {
        option_id: PermissionOptionId::from_string(id),
        name: name.to_string(),
        kind: Open::Known(kind),
        meta: None,
        extra: BTreeMap::default(),
    }
}

fn default_permission_options() -> Vec<PermissionOption> {
    vec![
        perm_option(
            OPT_ALLOW_ONCE,
            "Allow once",
            PermissionOptionKind::AllowOnce,
        ),
        perm_option(
            OPT_ALLOW_ALWAYS,
            "Allow for session",
            PermissionOptionKind::AllowAlways,
        ),
        perm_option(
            OPT_REJECT_ONCE,
            "Reject once",
            PermissionOptionKind::RejectOnce,
        ),
        perm_option(
            OPT_REJECT_ALWAYS,
            "Reject for session",
            PermissionOptionKind::RejectAlways,
        ),
    ]
}

/// [`ControlHandler`] implementation bound to one ACP session, forwarding
/// `can_use_tool` prompts over [`ClientPeer::request_permission`].
#[derive(Debug, Clone)]
pub struct SessionControlHandler {
    cx: SessionContext,
    memory: Arc<PermissionMemory>,
    mode: Arc<Mutex<AcpMode>>,
    turn_cancelled: Arc<AtomicBool>,
}

impl SessionControlHandler {
    pub(crate) const fn new(
        cx: SessionContext,
        memory: Arc<PermissionMemory>,
        mode: Arc<Mutex<AcpMode>>,
        turn_cancelled: Arc<AtomicBool>,
    ) -> Self {
        Self {
            cx,
            memory,
            mode,
            turn_cancelled,
        }
    }
}

impl ControlHandler for SessionControlHandler {
    #[allow(clippy::too_many_lines)]
    async fn can_use_tool(
        &self,
        req: ToolPermissionRequest,
    ) -> claude_acp_sdk::Result<PermissionDecision> {
        if let Some(decision) = self.memory.evaluate(&req) {
            return Ok(decision);
        }

        {
            let current_mode = *self.mode.lock().expect("session mode mutex poisoned");
            match current_mode {
                AcpMode::AgentFullAccess => return Ok(PermissionDecision::allow()),
                AcpMode::DontAsk => {
                    return Ok(PermissionDecision::deny(format!(
                        "tool `{}` denied in dont-ask mode",
                        req.tool_name
                    )));
                }
                AcpMode::ReadOnly | AcpMode::WorkspaceWrite | AcpMode::Agent | AcpMode::Auto => {}
            }
        }

        let tool_call_id = req
            .tool_use_id
            .clone()
            .unwrap_or_else(|| format!("perm-{}", uuid::Uuid::new_v4()));
        let tool_call = build_tool_call(
            &tool_call_id,
            &req.tool_name,
            &req.input,
            ToolCallStatus::Pending,
        );
        let title = tool_call
            .title
            .clone()
            .into_option()
            .unwrap_or_else(|| req.tool_name.clone());

        let options = default_permission_options();
        let params = PermissionRequest {
            session_id: SessionId::from_string(self.cx.session_id().as_str()),
            title,
            description: None,
            subject: Some(Open::Known(PermissionSubject::ToolCall(Box::new(
                ToolCallSubject {
                    tool_call,
                    meta: None,
                    extra: BTreeMap::default(),
                },
            )))),
            options: options.clone(),
            meta: None,
            extra: BTreeMap::default(),
        };

        let outcome = match self.cx.request_permission(params).await {
            Ok(outcome) => outcome,
            Err(err) => {
                return Ok(PermissionDecision::deny_and_interrupt(format!(
                    "permission request failed: {err}"
                )));
            }
        };

        match outcome {
            PermissionOutcome::Cancelled(_) | PermissionOutcome::Unknown(..) => {
                self.turn_cancelled.store(true, Ordering::SeqCst);
                Ok(PermissionDecision::deny_and_interrupt(
                    "permission request cancelled by client",
                ))
            }
            PermissionOutcome::Custom(..) => Ok(PermissionDecision::deny(
                "unrecognized permission outcome".to_string(),
            )),
            PermissionOutcome::Selected(selected) => {
                let kind = options
                    .iter()
                    .find(|opt| opt.option_id == selected.option_id)
                    .and_then(|opt| match &opt.kind {
                        Open::Known(k) => Some(*k),
                        _ => None,
                    });

                match kind {
                    Some(PermissionOptionKind::AllowOnce) => Ok(PermissionDecision::allow()),
                    Some(PermissionOptionKind::AllowAlways) => {
                        let (updated_perms, promoted_mode) = self.memory.record_allow_always(&req);
                        if let Some(new_mode) = promoted_mode {
                            {
                                let mut guard =
                                    self.mode.lock().expect("session mode mutex poisoned");
                                *guard = new_mode;
                            }
                            let _ = self
                                .cx
                                .update(SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
                                    current_mode_id: ConfigValueId::from_string(new_mode.as_str()),
                                    meta: None,
                                    extra: BTreeMap::default(),
                                }))
                                .await;
                        }
                        Ok(PermissionDecision::allow().with_permissions(updated_perms))
                    }
                    Some(PermissionOptionKind::RejectOnce) => Ok(PermissionDecision::deny(
                        format!("tool `{}` rejected by user", req.tool_name),
                    )),
                    Some(PermissionOptionKind::RejectAlways) => {
                        self.memory.record_reject_always(&req);
                        Ok(PermissionDecision::deny(format!(
                            "tool `{}` rejected for this session",
                            req.tool_name
                        )))
                    }
                    None => Ok(PermissionDecision::deny(format!(
                        "unrecognized permission option `{}`",
                        selected.option_id.as_str()
                    ))),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn wildcard_matches_command_and_path_globs() {
        assert!(wildcard_match("cargo *", "cargo nextest run"));
        assert!(!wildcard_match("cargo *", "rm -rf /"));
        assert!(wildcard_match("/workspace/**", "/workspace/src/lib.rs"));
        assert!(!wildcard_match("/workspace/**", "/etc/passwd"));
    }

    #[test]
    fn permission_memory_enforces_deny_over_allow() {
        let mem = PermissionMemory::new();
        let req = ToolPermissionRequest {
            tool_name: "Bash".to_string(),
            input: json!({ "command": "cargo test" }),
            permission_suggestions: Vec::new(),
            blocked_path: None,
            decision_reason: None,
            tool_use_id: None,
        };
        let _ = mem.record_allow_always(&req);
        assert_eq!(mem.evaluate(&req), Some(PermissionDecision::allow()));

        mem.record_reject_always(&req);
        assert!(matches!(
            mem.evaluate(&req),
            Some(PermissionDecision::Deny {
                interrupt: false,
                ..
            })
        ));
    }
}
