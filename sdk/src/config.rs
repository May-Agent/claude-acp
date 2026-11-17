//! Session and child-process launch configuration for Claude Code.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use claude_codes::io::McpServerConfig;
use claude_codes::{EffortLevel, InitPermissionMode};
use serde::Serialize;
use uuid::Uuid;

use crate::error::{Result, SdkError};

/// Session permission mode accepted by `--permission-mode` and
/// `set_permission_mode`.
pub type PermissionMode = InitPermissionMode;

/// Alias for [`SessionConfig`] used when spawning a Claude CLI process.
pub type SpawnConfig = SessionConfig;

/// Configuration for launching and driving a Claude Code `stream-json` session.
#[derive(Debug, Clone, Default)]
pub struct SessionConfig {
    /// Explicit path to the `claude` executable; falls back to `CLAUDE_PATH`,
    /// then `"claude"`.
    pub claude_path: Option<PathBuf>,
    /// Working directory for the spawned CLI process.
    pub cwd: Option<PathBuf>,
    /// Extra workspace directories granted via `--add-dir`.
    pub additional_directories: Vec<PathBuf>,
    /// Primary model identifier or alias (`--model`).
    pub model: Option<String>,
    /// Overload fallback model (`--fallback-model`).
    pub fallback_model: Option<String>,
    /// Initial session permission mode (`--permission-mode`).
    pub permission_mode: Option<PermissionMode>,
    /// Initial reasoning effort level (`--effort`).
    pub effort: Option<EffortLevel>,
    /// Whether to emit incremental `stream_event` chunks (`--include-partial-messages`).
    pub include_partial_messages: bool,
    /// Existing session ID to resume (`--resume`).
    pub resume: Option<String>,
    /// Create a fresh session ID when resuming (`--fork-session`).
    pub fork_session: bool,
    /// Explicit UUID for the session (`--session-id`).
    pub session_id: Option<Uuid>,
    /// Custom system prompt replacing the default (`--system-prompt`).
    pub system_prompt: Option<String>,
    /// Text appended to the default system prompt (`--append-system-prompt`).
    pub append_system_prompt: Option<String>,
    /// MCP server definitions rendered deterministically to `--mcp-config`.
    pub mcp_servers: BTreeMap<String, McpServerConfig>,
    /// Allowed tool patterns (`--allowedTools`).
    pub allowed_tools: Vec<String>,
    /// Disallowed tool patterns (`--disallowedTools`).
    pub disallowed_tools: Vec<String>,
    /// Extended-thinking token ceiling (`--max-thinking-tokens`).
    pub max_thinking_tokens: Option<u32>,
    /// Extra raw CLI arguments appended deterministically at the end of `argv`.
    pub extra_args: Vec<String>,
    /// Environment variables passed to the spawned CLI process.
    pub env: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct McpConfigEnvelope<'a> {
    #[serde(rename = "mcpServers")]
    mcp_servers: &'a BTreeMap<String, McpServerConfig>,
}

impl SessionConfig {
    /// Construct a default [`SessionConfig`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the path to the `claude` executable.
    #[must_use]
    pub fn claude_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.claude_path = Some(path.into());
        self
    }

    /// Set the working directory for the spawned CLI process.
    #[must_use]
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Add additional workspace directories (`--add-dir`).
    #[must_use]
    pub fn additional_directories<I, P>(mut self, dirs: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        self.additional_directories
            .extend(dirs.into_iter().map(Into::into));
        self
    }

    /// Set the session model (`--model`), accepting [`claude_codes::ClaudeModel`]
    /// or any string slice.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the fallback model (`--fallback-model`).
    #[must_use]
    pub fn fallback_model(mut self, model: impl Into<String>) -> Self {
        self.fallback_model = Some(model.into());
        self
    }

    /// Set the initial permission mode (`--permission-mode`).
    #[must_use]
    pub fn permission_mode(mut self, mode: PermissionMode) -> Self {
        self.permission_mode = Some(mode);
        self
    }

    /// Set the initial reasoning effort level (`--effort`).
    #[must_use]
    pub fn effort(mut self, effort: EffortLevel) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Enable or disable partial streaming deltas (`--include-partial-messages`).
    #[must_use]
    pub const fn include_partial_messages(mut self, enabled: bool) -> Self {
        self.include_partial_messages = enabled;
        self
    }

    /// Resume a persisted session by ID (`--resume`).
    #[must_use]
    pub fn resume(mut self, session_id: impl Into<String>) -> Self {
        self.resume = Some(session_id.into());
        self
    }

    /// Fork into a new session when resuming (`--fork-session`).
    #[must_use]
    pub const fn fork_session(mut self, fork: bool) -> Self {
        self.fork_session = fork;
        self
    }

    /// Pin an explicit session UUID (`--session-id`).
    #[must_use]
    pub const fn session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set the full system prompt (`--system-prompt`).
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Append text to the default system prompt (`--append-system-prompt`).
    #[must_use]
    pub fn append_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.append_system_prompt = Some(prompt.into());
        self
    }

    /// Register an MCP server definition for `--mcp-config`.
    #[must_use]
    pub fn mcp_server(mut self, name: impl Into<String>, config: McpServerConfig) -> Self {
        self.mcp_servers.insert(name.into(), config);
        self
    }

    /// Register multiple MCP server definitions for `--mcp-config`.
    #[must_use]
    pub fn mcp_servers<I, K>(mut self, servers: I) -> Self
    where
        I: IntoIterator<Item = (K, McpServerConfig)>,
        K: Into<String>,
    {
        for (name, cfg) in servers {
            self.mcp_servers.insert(name.into(), cfg);
        }
        self
    }

    /// Add allowed tool patterns (`--allowedTools`).
    #[must_use]
    pub fn allowed_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools.extend(tools.into_iter().map(Into::into));
        self
    }

    /// Add disallowed tool patterns (`--disallowedTools`).
    #[must_use]
    pub fn disallowed_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.disallowed_tools
            .extend(tools.into_iter().map(Into::into));
        self
    }

    /// Set the extended-thinking token budget (`--max-thinking-tokens`).
    #[must_use]
    pub const fn max_thinking_tokens(mut self, tokens: u32) -> Self {
        self.max_thinking_tokens = Some(tokens);
        self
    }

    /// Append raw CLI flags (`extra_args`).
    #[must_use]
    pub fn extra_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Insert an environment variable for the spawned process.
    #[must_use]
    pub fn env_var(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    /// Insert multiple environment variables for the spawned process.
    #[must_use]
    pub fn envs<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        for (key, value) in vars {
            self.env.insert(key.into(), value.into());
        }
        self
    }

    /// Resolve the executable path: `self.claude_path`, else non-empty
    /// `CLAUDE_PATH` from the process environment, else `"claude"`.
    #[must_use]
    pub fn resolve_executable(&self) -> PathBuf {
        self.resolve_executable_with(|key| std::env::var_os(key))
    }

    fn resolve_executable_with<F>(&self, lookup_env: F) -> PathBuf
    where
        F: FnOnce(&str) -> Option<OsString>,
    {
        if let Some(path) = &self.claude_path {
            return path.clone();
        }
        if let Some(env_val) = lookup_env("CLAUDE_PATH")
            && !env_val.is_empty()
        {
            return PathBuf::from(env_val);
        }
        PathBuf::from("claude")
    }

    /// Render the deterministic `--mcp-config` JSON string, or `Ok(None)` when
    /// no MCP servers are configured.
    ///
    /// Nested maps (such as `env` and `headers`) are canonicalized through
    /// `serde_json::Value` (`BTreeMap`-backed) so key order is deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Json`] if serialization fails.
    pub fn mcp_config_json(&self) -> Result<Option<String>> {
        if self.mcp_servers.is_empty() {
            return Ok(None);
        }
        let canonical = serde_json::to_value(McpConfigEnvelope {
            mcp_servers: &self.mcp_servers,
        })
        .map_err(|err| SdkError::json("mcp_config", err))?;
        serde_json::to_string(&canonical)
            .map(Some)
            .map_err(|err| SdkError::json("mcp_config", err))
    }

    /// Render the complete deterministic CLI argument vector for `claude`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Json`] if `--mcp-config` payload serialization fails.
    pub fn build_args(&self) -> Result<Vec<String>> {
        let mut args = vec![
            "--print".to_string(),
            "--output-format".to_string(),
            "stream-json".to_string(),
            "--input-format".to_string(),
            "stream-json".to_string(),
            "--verbose".to_string(),
        ];

        if self.include_partial_messages {
            args.push("--include-partial-messages".to_string());
        }
        if let Some(resume) = &self.resume {
            args.push("--resume".to_string());
            args.push(resume.clone());
        }
        if self.fork_session {
            args.push("--fork-session".to_string());
        }
        if let Some(session_id) = &self.session_id {
            args.push("--session-id".to_string());
            args.push(session_id.to_string());
        }
        if let Some(mode) = &self.permission_mode {
            args.push("--permission-mode".to_string());
            args.push(mode.as_str().to_string());
        }
        if let Some(model) = &self.model {
            args.push("--model".to_string());
            args.push(model.clone());
        }
        if let Some(fallback) = &self.fallback_model {
            args.push("--fallback-model".to_string());
            args.push(fallback.clone());
        }
        if let Some(effort) = &self.effort {
            args.push("--effort".to_string());
            args.push(effort.as_str().to_string());
        }
        if let Some(tokens) = self.max_thinking_tokens {
            args.push("--max-thinking-tokens".to_string());
            args.push(tokens.to_string());
        }
        if let Some(prompt) = &self.system_prompt {
            args.push("--system-prompt".to_string());
            args.push(prompt.clone());
        }
        if let Some(append) = &self.append_system_prompt {
            args.push("--append-system-prompt".to_string());
            args.push(append.clone());
        }
        if !self.additional_directories.is_empty() {
            args.push("--add-dir".to_string());
            args.extend(
                self.additional_directories
                    .iter()
                    .map(|p| path_to_arg(p.as_path())),
            );
        }
        if let Some(mcp_json) = self.mcp_config_json()? {
            args.push("--mcp-config".to_string());
            args.push(mcp_json);
        }
        if !self.allowed_tools.is_empty() {
            args.push("--allowedTools".to_string());
            args.extend(self.allowed_tools.iter().cloned());
        }
        if !self.disallowed_tools.is_empty() {
            args.push("--disallowedTools".to_string());
            args.extend(self.disallowed_tools.iter().cloned());
        }

        // Emit `--permission-prompt-tool stdio` after variadic flags so
        // trailing `extra_args` cannot be consumed as variadic values.
        args.push("--permission-prompt-tool".to_string());
        args.push("stdio".to_string());

        args.extend(self.extra_args.iter().cloned());
        Ok(args)
    }
}

fn path_to_arg(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use claude_codes::ClaudeModel;
    use claude_codes::io::{McpHttpServerConfig, McpStdioServerConfig};

    use super::*;

    #[test]
    fn resolves_executable_precedence() {
        let explicit = SessionConfig::new().claude_path("/opt/bin/claude-custom");
        assert_eq!(
            explicit.resolve_executable_with(|_| Some(OsString::from("/env/claude"))),
            PathBuf::from("/opt/bin/claude-custom")
        );

        let from_env = SessionConfig::new();
        assert_eq!(
            from_env.resolve_executable_with(|_| Some(OsString::from("/env/claude"))),
            PathBuf::from("/env/claude")
        );

        assert_eq!(
            from_env.resolve_executable_with(|_| Some(OsString::new())),
            PathBuf::from("claude")
        );
        assert_eq!(
            from_env.resolve_executable_with(|_| None),
            PathBuf::from("claude")
        );
    }

    #[test]
    fn builds_deterministic_cli_args_and_mcp_json() {
        let sid = Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let mut stdio_env = HashMap::new();
        stdio_env.insert("Z_VAR".to_string(), "last".to_string());
        stdio_env.insert("A_VAR".to_string(), "first".to_string());

        let cfg = SpawnConfig::new()
            .cwd("/workspace/project")
            .additional_directories(["/workspace/extra-a", "/workspace/extra-b"])
            .model(ClaudeModel::Sonnet)
            .fallback_model("opus")
            .permission_mode(PermissionMode::AcceptEdits)
            .effort(EffortLevel::High)
            .include_partial_messages(true)
            .resume("prev-session")
            .fork_session(true)
            .session_id(sid)
            .system_prompt("sys")
            .append_system_prompt("append")
            .mcp_server(
                "zeta",
                McpServerConfig::Http(McpHttpServerConfig {
                    url: "https://example.invalid/mcp".to_string(),
                    headers: None,
                }),
            )
            .mcp_server(
                "alpha",
                McpServerConfig::Stdio(McpStdioServerConfig {
                    command: "npx".to_string(),
                    args: Some(vec!["-y".to_string(), "srv".to_string()]),
                    env: Some(stdio_env),
                }),
            )
            .allowed_tools(["Read", "Bash(git *)"])
            .disallowed_tools(["WebFetch"])
            .max_thinking_tokens(8192)
            .extra_args(["--strict-mcp-config"])
            .env_var("FOO", "bar");

        let mcp_json = cfg.mcp_config_json().unwrap().unwrap();
        assert_eq!(
            mcp_json,
            r#"{"mcpServers":{"alpha":{"args":["-y","srv"],"command":"npx","env":{"A_VAR":"first","Z_VAR":"last"},"type":"stdio"},"zeta":{"type":"http","url":"https://example.invalid/mcp"}}}"#
        );

        let args = cfg.build_args().unwrap();
        assert_eq!(
            args,
            vec![
                "--print",
                "--output-format",
                "stream-json",
                "--input-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--resume",
                "prev-session",
                "--fork-session",
                "--session-id",
                "550e8400-e29b-41d4-a716-446655440000",
                "--permission-mode",
                "acceptEdits",
                "--model",
                "sonnet",
                "--fallback-model",
                "opus",
                "--effort",
                "high",
                "--max-thinking-tokens",
                "8192",
                "--system-prompt",
                "sys",
                "--append-system-prompt",
                "append",
                "--add-dir",
                "/workspace/extra-a",
                "/workspace/extra-b",
                "--mcp-config",
                &mcp_json,
                "--allowedTools",
                "Read",
                "Bash(git *)",
                "--disallowedTools",
                "WebFetch",
                "--permission-prompt-tool",
                "stdio",
                "--strict-mcp-config",
            ]
        );
    }
}
