# claude-acp

Agent Client Protocol (ACP v1) server backed by Claude Code (`claude-acp-sdk`) and built on `aither-acp`.

## Architecture

The workspace is split into two runtime-agnostic crates:

- **`claude-acp-sdk`** (`sdk/`) — LF-delimited `stream-json` NDJSON transport, process launcher (`process` feature), concurrent bidirectional control-protocol router (`initialize`, `interrupt`, `set_permission_mode`, `set_model`, `set_max_thinking_tokens`, `apply_flag_settings`, `can_use_tool`, `hook_callback`, `mcp_message`), and non-blocking `~/.claude/projects` transcript store.
- **`claude-acp`** (`src/`) — `aither_acp::AgentHandler` implementation (`ClaudeAgent`) bridging ACP v1 JSON-RPC traffic, streaming updates, tool diffs, permission dialogs, session lifecycle, modes, config options, and slash commands over any `BidirectionalTransport` (`serve`) or stdio (`serve_stdio`).

Neither library crate depends on `tokio`; callers bring their own executor.

## Capabilities

During `initialize`, `ClaudeAgent` advertises ACP protocol version `1` with:

- **Prompt**: `image: true`, `audio: false`, `embeddedContext: true`
- **MCP**: `stdio` plus `http: true` and `sse: true` (forwarded deterministically via `--mcp-config`)
- **Sessions**: `loadSession: true`, `list`, `delete`, `close`, `resume`, `additionalDirectories`
- **Authentication**: `api-key` and `claude-login`, `auth.logout`, and `_meta.authStatus` (`_auth/status_update` extension notifications)

## Usage

### Stdio ACP server

```rust,no_run
use claude_acp::{ClaudeAcpConfig, serve_stdio};

#[tokio::main]
async fn main() {
    let config = ClaudeAcpConfig::new().default_mode("workspace-write");
    let (_peer, connection) = serve_stdio(config);
    connection.await;
}
```

### Embedded duplex transport

```rust,no_run
use aither_acp::{AcpClient, ClientHandler, PromptParams, SessionNewParams, SessionNotification};
use aither_acp::{RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult};
use aither_mcp::protocol::JsonRpcError;
use aither_mcp::transport::DuplexTransport;
use claude_acp::{ClaudeAcpConfig, serve};
use std::future::Future;

struct EditorHandler;

impl ClientHandler for EditorHandler {
    fn session_update(&self, _update: SessionNotification) -> impl Future<Output = ()> + Send {
        async {}
    }

    fn request_permission(
        &self,
        _params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        async {
            Ok(RequestPermissionResult {
                outcome: RequestPermissionOutcome::Cancelled,
                meta: None,
            })
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (client_transport, server_transport) = DuplexTransport::pair();
    let (_peer, server_conn) = serve(server_transport, ClaudeAcpConfig::new());
    let (client, client_conn) = AcpClient::connect(client_transport, EditorHandler);

    tokio::spawn(server_conn);
    tokio::spawn(client_conn);

    let _init = client.initialize().await?;
    let session = client.new_session(SessionNewParams::new("/tmp")).await?;
    let _ = PromptParams::new(&session.session_id, vec![]);
    Ok(())
}
```

## Configuration

`ClaudeAcpConfig` controls session launch defaults and server behavior:

| Builder / Field | Description |
| --- | --- |
| `spawn` (`SpawnConfig`) | Base CLI configuration (`claude_path`, `model`, `effort`, `env`, `allowed_tools`, `extra_args`, etc.) |
| `default_mode` | Initial ACP mode (`"read-only"`, `"workspace-write"`, `"agent"`, `"agent-full-access"`) |
| `include_partial_messages` | Stream incremental token and thinking deltas (`true` by default) |
| `transcript_home` | Override `<home>/.claude/projects` base path for isolated transcript storage |
| `session_list_page_size` | Maximum entries returned per `session/list` page before emitting `nextCursor` |
| `connector` | Custom `SessionConnector` for attaching in-memory streams without spawning `claude` |

### Modes (`session/set_mode` & `mode` config option)

| ACP Mode ID | Claude Permission Mode | Behavior |
| --- | --- | --- |
| `read-only` | `plan` | Read/search analysis and planning without file edits or shell execution |
| `workspace-write` | `acceptEdits` | Auto-accept file edits while prompting for shell/external tools |
| `agent` | `default` | Interactive permission approval via `session/request_permission` |
| `agent-full-access` | `bypassPermissions` | Bypass permission prompts for all tool invocations |

### Config Options (`session/set_config_option`)

| Option ID | Category | Accepted Values |
| --- | --- | --- |
| `mode` | `mode` | `read-only`, `workspace-write`, `agent`, `agent-full-access` |
| `model` | `model` | Any advertised CLI model or `ClaudeModel` alias (`sonnet`, `opus`, `haiku`, `opusplan`, `sonnet[1m]`, ...) |
| `reasoning_effort` | `thought_level` | `off`, `low`, `medium`, `high`, `xhigh`, `max`, or a positive max-thinking-token integer |

### Slash Commands

| Command | Handling |
| --- | --- |
| `/compact [instructions]` | Compacts conversation history in the active CLI session |
| `/clear` | Clears conversation history and resets session permission memory |
| `/status` | Reports active session ID, working directory, mode, model, effort, auth status, and MCP servers |
| `/logout` | Clears active session credentials and pushes `_auth/status_update` (`none`) |
| `/review [target]` | Runs native CLI `/review` when advertised or dispatches a code-review prompt |

## Cargo Features

- `process` *(default)* — Enables `claude-acp-sdk/process` and `ProcessSessionConnector` to launch local `claude` child processes. Disable default features (`--no-default-features`) when embedding exclusively over custom `SessionConnector` streams.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
