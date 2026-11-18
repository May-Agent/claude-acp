//! Authentication methods (`api-key`, `claude-login`), credential state, and
//! `_auth/status_update` extension notifications.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use aither_acp::ext::auth_status::{
    self, AuthStatus, AuthStatusAccount, AuthStatusKind, CAPABILITY, METHOD,
};
use aither_acp::{
    AgentAuthCapabilities, AgentCapabilities, AuthMethod, AuthMethodAgent, AuthMethodTerminal,
    AuthenticateParams, ClientPeer, LogoutCapabilities,
};
use claude_acp_sdk::{AuthStatusMessage, SpawnConfig};
use serde_json::{Value, json};

use crate::error::{ClaudeAcpError, Result};

/// Auth method identifier for Anthropic API key authentication.
pub const AUTH_METHOD_API_KEY: &str = "api-key";

/// Auth method identifier for interactive Claude account login.
pub const AUTH_METHOD_CLAUDE_LOGIN: &str = "claude-login";

const VENDOR_ANTHROPIC: &str = "Anthropic";

/// Attach `auth` (`logout: {}`) and `_meta.authStatus: {}` capabilities onto
/// [`AgentCapabilities`].
pub fn apply_auth_capabilities(caps: &mut AgentCapabilities) {
    caps.auth = Some(AgentAuthCapabilities {
        logout: Some(LogoutCapabilities::default()),
        meta: None,
    });
    let mut meta_obj = caps
        .meta
        .take()
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default();
    meta_obj.insert(CAPABILITY.to_string(), json!({}));
    caps.meta = Some(Value::Object(meta_obj));
}

/// Build the list of [`AuthMethod`]s advertised in `initialize`.
///
/// When the client advertises `clientCapabilities.auth.terminal == true`,
/// `claude-login` is exposed as a terminal auth method invoking
/// `<claude> auth login`; otherwise it is exposed as an agent-managed method.
pub fn build_auth_methods(
    client_supports_terminal_auth: bool,
    claude_executable: &Path,
) -> Vec<AuthMethod> {
    let api_key = AuthMethod::Agent(
        AuthMethodAgent::new(AUTH_METHOD_API_KEY, "Anthropic API Key")
            .description("Authenticate using an Anthropic API key"),
    );

    let login = if client_supports_terminal_auth {
        AuthMethod::Terminal(AuthMethodTerminal {
            id: AUTH_METHOD_CLAUDE_LOGIN.to_string(),
            name: "Claude Login".to_string(),
            description: Some("Sign in interactively with your Claude account".to_string()),
            args: vec![
                claude_executable.to_string_lossy().into_owned(),
                "auth".to_string(),
                "login".to_string(),
            ],
            env: HashMap::new(),
            meta: None,
        })
    } else {
        AuthMethod::Agent(
            AuthMethodAgent::new(AUTH_METHOD_CLAUDE_LOGIN, "Claude Login")
                .description("Sign in with your Claude account"),
        )
    };

    vec![api_key, login]
}

/// Mutable connection-level authentication state managed by `ClaudeAgent`.
#[derive(Debug, Clone)]
pub struct AuthState {
    api_key: Option<String>,
    logged_out: bool,
    status: AuthStatus,
}

impl AuthState {
    /// Initialize [`AuthState`] from the base [`SpawnConfig`] environment.
    pub(crate) fn from_spawn(spawn: &SpawnConfig) -> Self {
        if let Some(configured_key) = spawn
            .env
            .get("ANTHROPIC_API_KEY")
            .filter(|v| !v.trim().is_empty())
        {
            return Self {
                api_key: Some(configured_key.clone()),
                logged_out: false,
                status: build_status(
                    AuthStatusKind::ApiKey,
                    Some("Anthropic API Key"),
                    None,
                    None,
                ),
            };
        }

        if std::env::var_os("ANTHROPIC_API_KEY").is_some_and(|v| !v.is_empty()) {
            return Self {
                api_key: None,
                logged_out: false,
                status: build_status(
                    AuthStatusKind::External,
                    Some("ANTHROPIC_API_KEY"),
                    None,
                    None,
                ),
            };
        }

        Self {
            api_key: None,
            logged_out: false,
            status: build_status(AuthStatusKind::None, None, None, None),
        }
    }

    /// Snapshot the current [`AuthStatus`].
    pub(crate) fn status(&self) -> AuthStatus {
        self.status.clone()
    }

    /// Apply active credential overrides onto a per-session [`SpawnConfig`].
    pub(crate) fn apply_to_spawn(&self, spawn: &mut SpawnConfig) {
        if let Some(key) = &self.api_key {
            spawn
                .env
                .insert("ANTHROPIC_API_KEY".to_string(), key.clone());
        } else if self.logged_out {
            spawn.env.remove("ANTHROPIC_API_KEY");
        }
    }

    /// Handle an ACP `authenticate` request and return the updated [`AuthStatus`].
    pub(crate) fn authenticate(&mut self, params: &AuthenticateParams) -> Result<AuthStatus> {
        match params.method_id.as_str() {
            AUTH_METHOD_API_KEY => {
                let supplied_key = params
                    .meta
                    .as_ref()
                    .and_then(Value::as_object)
                    .and_then(|obj| {
                        obj.get("apiKey")
                            .or_else(|| obj.get("api_key"))
                            .or_else(|| obj.get("token"))
                    })
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(ToOwned::to_owned);

                if let Some(key) = supplied_key {
                    self.api_key = Some(key);
                    self.logged_out = false;
                    self.status = build_status(
                        AuthStatusKind::ApiKey,
                        Some("Anthropic API Key"),
                        None,
                        None,
                    );
                    return Ok(self.status.clone());
                }

                let has_ambient_key = self.api_key.is_some()
                    || (!self.logged_out
                        && std::env::var_os("ANTHROPIC_API_KEY").is_some_and(|v| !v.is_empty()));

                if has_ambient_key {
                    self.logged_out = false;
                    self.status = build_status(
                        AuthStatusKind::ApiKey,
                        Some("Anthropic API Key"),
                        None,
                        None,
                    );
                    Ok(self.status.clone())
                } else {
                    Err(ClaudeAcpError::InvalidParams(
                        "API key required in `_meta.apiKey` or `ANTHROPIC_API_KEY`".to_string(),
                    ))
                }
            }
            AUTH_METHOD_CLAUDE_LOGIN => {
                let meta_obj = params.meta.as_ref().and_then(Value::as_object);
                let email = meta_field(meta_obj, "email");
                let organization = meta_field(meta_obj, "organization")
                    .or_else(|| meta_field(meta_obj, "orgName"));
                let plan = meta_field(meta_obj, "plan")
                    .or_else(|| meta_field(meta_obj, "subscriptionType"));

                let label_owned = email
                    .clone()
                    .unwrap_or_else(|| "Claude Account".to_string());
                let account = if email.is_some() || organization.is_some() || plan.is_some() {
                    Some(AuthStatusAccount {
                        email,
                        organization,
                        plan,
                        extra: BTreeMap::new(),
                    })
                } else {
                    None
                };

                self.api_key = None;
                self.logged_out = false;
                self.status =
                    build_status(AuthStatusKind::Account, Some(&label_owned), None, account);
                Ok(self.status.clone())
            }
            other => Err(ClaudeAcpError::InvalidParams(format!(
                "unsupported auth method `{other}`; expected `{AUTH_METHOD_API_KEY}` or `{AUTH_METHOD_CLAUDE_LOGIN}`"
            ))),
        }
    }

    /// Clear credentials and mark the agent signed out.
    pub(crate) fn logout(&mut self) -> AuthStatus {
        self.api_key = None;
        self.logged_out = true;
        self.status = build_status(AuthStatusKind::None, None, None, None);
        self.status.clone()
    }

    /// Translate an inbound CLI [`AuthStatusMessage`] into an updated [`AuthStatus`].
    pub(crate) fn observe_cli_auth_status(&mut self, msg: &AuthStatusMessage) -> AuthStatus {
        let detail_joined = (!msg.output.is_empty()).then(|| msg.output.join("\n"));
        if let Some(err) = &msg.error {
            self.status = build_status(AuthStatusKind::None, Some("Auth error"), Some(err), None);
        } else if msg.is_authenticating {
            let kind = if self.api_key.is_some() {
                AuthStatusKind::ApiKey
            } else {
                AuthStatusKind::Account
            };
            self.status = build_status(
                kind,
                Some("Authenticating"),
                detail_joined.as_deref(),
                self.status.account.clone(),
            );
        } else {
            self.logged_out = false;
            let kind = if self.api_key.is_some() {
                AuthStatusKind::ApiKey
            } else if matches!(self.status.kind, AuthStatusKind::None) {
                AuthStatusKind::Account
            } else {
                self.status.kind.clone()
            };
            let label = self
                .status
                .label
                .clone()
                .unwrap_or_else(|| "Authenticated".to_string());
            self.status = build_status(
                kind,
                Some(&label),
                detail_joined.as_deref(),
                self.status.account.clone(),
            );
        }
        self.status.clone()
    }
}

/// Push an `_auth/status_update` notification to the connected ACP client.
pub async fn notify_auth_status(peer: &ClientPeer, status: &AuthStatus) -> Result<()> {
    let payload = json!({
        auth_status::CAPABILITY: status,
    });
    peer.ext_notify(METHOD, Some(payload)).await?;
    Ok(())
}

fn meta_field(obj: Option<&serde_json::Map<String, Value>>, key: &str) -> Option<String> {
    obj?.get(key)?
        .as_str()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToOwned::to_owned)
}

fn build_status(
    kind: AuthStatusKind,
    label: Option<&str>,
    detail: Option<&str>,
    account: Option<AuthStatusAccount>,
) -> AuthStatus {
    AuthStatus {
        kind,
        label: label.map(ToOwned::to_owned),
        detail: detail.map(ToOwned::to_owned),
        account,
        vendor: Some(VENDOR_ANTHROPIC.to_string()),
        extra: BTreeMap::new(),
    }
}
