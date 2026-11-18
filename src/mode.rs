//! Session modes, configuration options (`mode`, `model`, `reasoning_effort`),
//! and advertised slash commands.

use aither_acp::{
    AvailableCommand, AvailableCommandInput, ConfigOption, ConfigOptionValue, ConfigSelectOption,
    ConfigSelectOptions, SessionMode, SessionModeState,
};
use claude_acp_sdk::{ClaudeModel, CommandInfo, EffortLevel, PermissionMode};
use serde_json::Value;

use crate::error::{ClaudeAcpError, Result};

/// Canonical ACP session mode identifiers supported by `claude-acp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcpMode {
    /// Read-only analysis and planning (`PermissionMode::Plan`).
    ReadOnly,
    /// Auto-accept workspace file edits (`PermissionMode::AcceptEdits`).
    WorkspaceWrite,
    /// Standard interactive agent prompting (`PermissionMode::Default`).
    #[default]
    Agent,
    /// Unprompted full tool access (`PermissionMode::BypassPermissions`).
    AgentFullAccess,
}

impl AcpMode {
    /// Stable wire identifier for this mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ReadOnly => "read-only",
            Self::WorkspaceWrite => "workspace-write",
            Self::Agent => "agent",
            Self::AgentFullAccess => "agent-full-access",
        }
    }

    /// Human-readable display label.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::ReadOnly => "Read Only",
            Self::WorkspaceWrite => "Workspace Write",
            Self::Agent => "Agent",
            Self::AgentFullAccess => "Agent (Full Access)",
        }
    }

    /// Human-readable mode description.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::ReadOnly => "Analyze and plan without modifying files or running commands",
            Self::WorkspaceWrite => "Automatically accept file edits within the workspace",
            Self::Agent => "Prompt for approval before modifying files or running commands",
            Self::AgentFullAccess => "Bypass permission prompts for all tool executions",
        }
    }

    /// All supported [`AcpMode`] variants in display order.
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [
            Self::ReadOnly,
            Self::WorkspaceWrite,
            Self::Agent,
            Self::AgentFullAccess,
        ]
    }

    /// Parse an ACP mode identifier string.
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeAcpError::InvalidParams`] if `mode_id` is unrecognized.
    pub fn parse(mode_id: &str) -> Result<Self> {
        match mode_id {
            "read-only" => Ok(Self::ReadOnly),
            "workspace-write" => Ok(Self::WorkspaceWrite),
            "agent" => Ok(Self::Agent),
            "agent-full-access" => Ok(Self::AgentFullAccess),
            other => Err(ClaudeAcpError::InvalidParams(format!(
                "unsupported session mode `{other}`; expected `read-only`, `workspace-write`, `agent`, or `agent-full-access`"
            ))),
        }
    }

    /// Map this ACP mode onto the Claude CLI [`PermissionMode`].
    #[must_use]
    pub const fn to_permission_mode(self) -> PermissionMode {
        match self {
            Self::ReadOnly => PermissionMode::Plan,
            Self::WorkspaceWrite => PermissionMode::AcceptEdits,
            Self::Agent => PermissionMode::Default,
            Self::AgentFullAccess => PermissionMode::BypassPermissions,
        }
    }

    /// Map a Claude CLI [`PermissionMode`] onto its corresponding [`AcpMode`].
    #[must_use]
    pub const fn from_permission_mode(mode: &PermissionMode) -> Self {
        match mode {
            PermissionMode::Plan => Self::ReadOnly,
            PermissionMode::AcceptEdits => Self::WorkspaceWrite,
            PermissionMode::BypassPermissions => Self::AgentFullAccess,
            PermissionMode::Default
            | PermissionMode::DontAsk
            | PermissionMode::Auto
            | PermissionMode::Unknown(_) => Self::Agent,
        }
    }

    /// Build the ACP [`SessionModeState`] with `self` as active.
    #[must_use]
    pub fn into_mode_state(self) -> SessionModeState {
        SessionModeState {
            current_mode_id: self.as_str().to_string(),
            available_modes: Self::all()
                .into_iter()
                .map(|m| SessionMode {
                    id: m.as_str().to_string(),
                    name: m.name().to_string(),
                    description: Some(m.description().to_string()),
                    meta: None,
                })
                .collect(),
            meta: None,
        }
    }
}

/// Session reasoning configuration backing the `reasoning_effort` config option.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningSetting {
    /// Extended thinking disabled (`max_thinking_tokens = 0`).
    Off,
    /// Named Claude effort level (`low`, `medium`, `high`, `xhigh`, `max`).
    Effort(EffortLevel),
    /// Explicit extended-thinking token budget (`max_thinking_tokens`).
    Tokens(u32),
}

impl Default for ReasoningSetting {
    fn default() -> Self {
        Self::Effort(EffortLevel::Medium)
    }
}

impl ReasoningSetting {
    pub(crate) fn from_spawn(
        effort: Option<&EffortLevel>,
        max_thinking_tokens: Option<u32>,
    ) -> Self {
        if max_thinking_tokens == Some(0) {
            return Self::Off;
        }
        if let Some(eff) = effort {
            return Self::Effort(eff.clone());
        }
        if let Some(tokens) = max_thinking_tokens {
            return Self::Tokens(tokens);
        }
        Self::default()
    }

    pub(crate) fn parse(raw: &str) -> Result<Self> {
        match raw {
            "off" | "none" | "0" => Ok(Self::Off),
            "low" => Ok(Self::Effort(EffortLevel::Low)),
            "medium" => Ok(Self::Effort(EffortLevel::Medium)),
            "high" => Ok(Self::Effort(EffortLevel::High)),
            "xhigh" => Ok(Self::Effort(EffortLevel::Xhigh)),
            "max" => Ok(Self::Effort(EffortLevel::Max)),
            other => {
                if let Ok(tokens) = other.parse::<u32>()
                    && tokens > 0
                {
                    return Ok(Self::Tokens(tokens));
                }
                Err(ClaudeAcpError::InvalidParams(format!(
                    "unsupported `reasoning_effort` value `{other}`; expected `off`, `low`, `medium`, `high`, `xhigh`, `max`, or a positive token integer"
                )))
            }
        }
    }

    pub(crate) fn current_value_id(&self) -> String {
        match self {
            Self::Off => "off".to_string(),
            Self::Effort(level) => level.as_str().to_string(),
            Self::Tokens(tokens) => tokens.to_string(),
        }
    }
}

/// Build the full session [`ConfigOption`] list (`mode`, `model`, `reasoning_effort`).
pub fn build_config_options(
    mode: AcpMode,
    current_model: &str,
    available_models: &[ConfigSelectOption],
    reasoning: &ReasoningSetting,
) -> Vec<ConfigOption> {
    let mode_options = AcpMode::all()
        .into_iter()
        .map(|m| ConfigSelectOption {
            value: m.as_str().to_string(),
            name: m.name().to_string(),
            description: Some(m.description().to_string()),
            meta: None,
        })
        .collect();

    let mut model_choices = available_models.to_vec();
    if !model_choices.iter().any(|opt| opt.value == current_model) {
        let model_enum = ClaudeModel::from(current_model);
        model_choices.insert(
            0,
            ConfigSelectOption {
                value: current_model.to_string(),
                name: model_enum.display_name().to_string(),
                description: None,
                meta: None,
            },
        );
    }

    let effort_options = build_effort_options(reasoning);

    vec![
        ConfigOption {
            id: "mode".to_string(),
            name: "Mode".to_string(),
            description: Some("Session tool approval mode".to_string()),
            category: Some("mode".to_string()),
            kind: Some("select".to_string()),
            current_value: Some(ConfigOptionValue::Selected(mode.as_str().to_string())),
            options: Some(ConfigSelectOptions::Flat(mode_options)),
            meta: None,
        },
        ConfigOption {
            id: "model".to_string(),
            name: "Model".to_string(),
            description: Some("Active Claude model".to_string()),
            category: Some("model".to_string()),
            kind: Some("select".to_string()),
            current_value: Some(ConfigOptionValue::Selected(current_model.to_string())),
            options: Some(ConfigSelectOptions::Flat(model_choices)),
            meta: None,
        },
        ConfigOption {
            id: "reasoning_effort".to_string(),
            name: "Reasoning Effort".to_string(),
            description: Some("Extended thinking effort level or token budget".to_string()),
            category: Some("thought_level".to_string()),
            kind: Some("select".to_string()),
            current_value: Some(ConfigOptionValue::Selected(reasoning.current_value_id())),
            options: Some(ConfigSelectOptions::Flat(effort_options)),
            meta: None,
        },
    ]
}

fn build_effort_options(reasoning: &ReasoningSetting) -> Vec<ConfigSelectOption> {
    let mut effort_options = vec![
        ConfigSelectOption {
            value: "off".to_string(),
            name: "Off".to_string(),
            description: Some("Disable extended thinking".to_string()),
            meta: None,
        },
        ConfigSelectOption {
            value: "low".to_string(),
            name: "Low".to_string(),
            description: Some("Fast responses with minimal thinking".to_string()),
            meta: None,
        },
        ConfigSelectOption {
            value: "medium".to_string(),
            name: "Medium".to_string(),
            description: Some("Balanced reasoning effort".to_string()),
            meta: None,
        },
        ConfigSelectOption {
            value: "high".to_string(),
            name: "High".to_string(),
            description: Some("Thorough reasoning for complex tasks".to_string()),
            meta: None,
        },
        ConfigSelectOption {
            value: "xhigh".to_string(),
            name: "Extra High".to_string(),
            description: Some("Extended deep reasoning (`ultrathink`)".to_string()),
            meta: None,
        },
        ConfigSelectOption {
            value: "max".to_string(),
            name: "Max".to_string(),
            description: Some("Maximum reasoning budget".to_string()),
            meta: None,
        },
    ];
    if let ReasoningSetting::Tokens(tokens) = reasoning {
        let token_str = tokens.to_string();
        if !effort_options.iter().any(|opt| opt.value == token_str) {
            effort_options.push(ConfigSelectOption {
                value: token_str,
                name: format!("{tokens} tokens"),
                description: Some("Custom max thinking token ceiling".to_string()),
                meta: None,
            });
        }
    }
    effort_options
}

/// Extract selectable model options from the SDK `InitializeResponse.models`
/// list, falling back to curated [`ClaudeModel`] aliases when empty.
pub fn extract_model_options(cli_models: &[Value]) -> Vec<ConfigSelectOption> {
    let mut out = Vec::new();
    for item in cli_models {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(val) = obj
            .get("value")
            .or_else(|| obj.get("id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };
        if out
            .iter()
            .any(|existing: &ConfigSelectOption| existing.value == val)
        {
            continue;
        }
        let name = obj
            .get("displayName")
            .or_else(|| obj.get("name"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map_or_else(
                || ClaudeModel::from(val).display_name().to_string(),
                ToOwned::to_owned,
            );
        let description = obj
            .get("description")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned);
        out.push(ConfigSelectOption {
            value: val.to_string(),
            name,
            description,
            meta: None,
        });
    }

    if out.is_empty() {
        for preset in [
            ClaudeModel::Sonnet,
            ClaudeModel::Opus,
            ClaudeModel::Haiku,
            ClaudeModel::OpusPlan,
            ClaudeModel::Sonnet1m,
            ClaudeModel::Opus1m,
        ] {
            out.push(ConfigSelectOption {
                value: preset.cli_arg().to_string(),
                name: preset.display_name().to_string(),
                description: None,
                meta: None,
            });
        }
    }

    out
}

/// Built-in slash commands always supported and intercepted by `claude-acp`,
/// merged with any additional commands advertised by the CLI on initialization.
pub fn build_available_commands(cli_commands: &[CommandInfo]) -> Vec<AvailableCommand> {
    let mut commands = vec![
        AvailableCommand {
            name: "compact".to_string(),
            description: "Compact conversation history to free context window space".to_string(),
            input: Some(AvailableCommandInput {
                hint: "Optional focus instructions".to_string(),
                meta: None,
            }),
            meta: None,
        },
        AvailableCommand {
            name: "clear".to_string(),
            description: "Clear conversation history and start fresh in this session".to_string(),
            input: None,
            meta: None,
        },
        AvailableCommand {
            name: "status".to_string(),
            description: "Show current session mode, model, effort, and auth status".to_string(),
            input: None,
            meta: None,
        },
        AvailableCommand {
            name: "logout".to_string(),
            description: "Clear active session credentials and sign out".to_string(),
            input: None,
            meta: None,
        },
        AvailableCommand {
            name: "review".to_string(),
            description: "Review workspace changes or a target path/branch for defects".to_string(),
            input: Some(AvailableCommandInput {
                hint: "Optional file, branch, or focus".to_string(),
                meta: None,
            }),
            meta: None,
        },
    ];

    for info in cli_commands {
        let normalized = info.name.trim_start_matches('/');
        if normalized.is_empty() || commands.iter().any(|c| c.name == normalized) {
            continue;
        }
        let input = (!info.argument_hint.trim().is_empty()).then(|| AvailableCommandInput {
            hint: info.argument_hint.clone(),
            meta: None,
        });
        commands.push(AvailableCommand {
            name: normalized.to_string(),
            description: info.description.clone(),
            input,
            meta: None,
        });
    }

    commands
}
