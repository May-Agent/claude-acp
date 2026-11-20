//! Session modes, configuration options (`mode`, `model`, `reasoning_effort`),
//! and advertised slash commands.

use std::collections::BTreeMap;

use aither_acp::ConfigId;
use aither_acp::domain::{
    AvailableCommand, Command, CommandInput, ConfigCategory, ConfigKind, ConfigOption,
    ConfigValueId, Open, SelectConfig, SelectOption, SelectOptions, TextInput,
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

fn config_option(
    id: &str,
    name: &str,
    description: &str,
    category: ConfigCategory,
    current_value: &str,
    options: Vec<SelectOption>,
) -> ConfigOption {
    ConfigOption {
        config_id: ConfigId::from_string(id),
        name: name.to_string(),
        description: Some(description.to_string()),
        category: Some(Open::Known(category)),
        kind: ConfigKind::Select(SelectConfig {
            current_value: ConfigValueId::from_string(current_value),
            options: SelectOptions::Ungrouped(options),
        }),
        meta: None,
        extra: BTreeMap::new(),
    }
}

fn select_option(value: &str, name: &str, description: Option<&str>) -> SelectOption {
    SelectOption {
        value: ConfigValueId::from_string(value),
        name: name.to_string(),
        description: description.map(ToOwned::to_owned),
        meta: None,
    }
}

/// Build the full session [`ConfigOption`] list (`mode`, `model`, `reasoning_effort`).
pub fn build_config_options(
    mode: AcpMode,
    current_model: &str,
    available_models: &[SelectOption],
    reasoning: &ReasoningSetting,
) -> Vec<ConfigOption> {
    let mode_options = AcpMode::all()
        .into_iter()
        .map(|m| select_option(m.as_str(), m.name(), Some(m.description())))
        .collect();

    let mut model_choices = available_models.to_vec();
    if !model_choices
        .iter()
        .any(|opt| opt.value.as_str() == current_model)
    {
        let model_enum = ClaudeModel::from(current_model);
        model_choices.insert(
            0,
            select_option(current_model, model_enum.display_name(), None),
        );
    }

    let effort_options = build_effort_options(reasoning);

    vec![
        config_option(
            "mode",
            "Mode",
            "Session tool approval mode",
            ConfigCategory::Mode,
            mode.as_str(),
            mode_options,
        ),
        config_option(
            "model",
            "Model",
            "Active Claude model",
            ConfigCategory::Model,
            current_model,
            model_choices,
        ),
        config_option(
            "reasoning_effort",
            "Reasoning Effort",
            "Extended thinking effort level or token budget",
            ConfigCategory::ThoughtLevel,
            &reasoning.current_value_id(),
            effort_options,
        ),
    ]
}

fn build_effort_options(reasoning: &ReasoningSetting) -> Vec<SelectOption> {
    let mut effort_options = vec![
        select_option("off", "Off", Some("Disable extended thinking")),
        select_option("low", "Low", Some("Fast responses with minimal thinking")),
        select_option("medium", "Medium", Some("Balanced reasoning effort")),
        select_option("high", "High", Some("Thorough reasoning for complex tasks")),
        select_option(
            "xhigh",
            "Extra High",
            Some("Extended deep reasoning (`ultrathink`)"),
        ),
        select_option("max", "Max", Some("Maximum reasoning budget")),
    ];
    if let ReasoningSetting::Tokens(tokens) = reasoning {
        let token_str = tokens.to_string();
        if !effort_options
            .iter()
            .any(|opt| opt.value.as_str() == token_str)
        {
            effort_options.push(SelectOption {
                value: ConfigValueId::from_string(token_str),
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
pub fn extract_model_options(cli_models: &[Value]) -> Vec<SelectOption> {
    let mut out: Vec<SelectOption> = Vec::new();
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
        if out.iter().any(|existing| existing.value.as_str() == val) {
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
        out.push(SelectOption {
            value: ConfigValueId::from_string(val),
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
            out.push(select_option(preset.cli_arg(), preset.display_name(), None));
        }
    }

    out
}

fn available_command(name: &str, description: &str, hint: Option<&str>) -> AvailableCommand {
    AvailableCommand::Command(Command {
        name: name.to_string(),
        description: description.to_string(),
        input: hint.map(|h| {
            Open::Known(CommandInput::Text(TextInput {
                hint: Some(h.to_string()),
                meta: None,
            }))
        }),
        meta: None,
        extra: BTreeMap::new(),
    })
}

/// Built-in slash commands always supported and intercepted by `claude-acp`,
/// merged with any additional commands advertised by the CLI on initialization.
pub fn build_available_commands(cli_commands: &[CommandInfo]) -> Vec<AvailableCommand> {
    let mut commands = vec![
        available_command(
            "compact",
            "Compact conversation history to free context window space",
            Some("Optional focus instructions"),
        ),
        available_command(
            "clear",
            "Clear conversation history and start fresh in this session",
            None,
        ),
        available_command(
            "status",
            "Show current session mode, model, effort, and auth status",
            None,
        ),
        available_command(
            "logout",
            "Clear active session credentials and sign out",
            None,
        ),
        available_command(
            "review",
            "Review workspace changes or a target path/branch for defects",
            Some("Optional file, branch, or focus"),
        ),
    ];

    for info in cli_commands {
        let normalized = info.name.trim_start_matches('/');
        if normalized.is_empty()
            || commands.iter().any(|c| match c {
                AvailableCommand::Command(cmd) => cmd.name == normalized,
                _ => false,
            })
        {
            continue;
        }
        let hint = (!info.argument_hint.trim().is_empty()).then_some(info.argument_hint.as_str());
        commands.push(available_command(normalized, &info.description, hint));
    }

    commands
}
