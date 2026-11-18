//! Bidirectional `stream-json` session client and concurrent protocol driver.

use std::collections::HashMap;
use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::Pin;
#[cfg(feature = "process")]
use std::process::ExitStatus;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::Poll;

use async_channel::{Receiver, Sender};
use claude_codes::io::{AssistantMessage, ResultMessage, UserMessage};
use claude_codes::{
    AnthropicError, ApplyFlagSettingsRequest, AuthStatusMessage, ClaudeInput, ClaudeOutput,
    CommandInfo, CommandLifecycleMessage, ContentBlock, ControlRequest, ControlRequestPayload,
    ControlResponse, ControlResponseMessage, ControlResponsePayload, ConversationResetMessage,
    EffortLevel, InitializeRequest, MessageRole, PromptSuggestionMessage, RateLimitEvent,
    StreamEventMessage, SystemMessage, ToolProgressMessage, ToolUseSummaryMessage,
};
use futures_lite::future::or;
use futures_lite::io::{AsyncRead, AsyncWrite};
#[cfg(feature = "process")]
use futures_lite::io::{AsyncReadExt, BufReader};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::config::PermissionMode;
#[cfg(feature = "process")]
use crate::config::SessionConfig;
use crate::error::{Result, SdkError};
use crate::handler::ControlHandler;
use crate::transport::{NdjsonReader, NdjsonWriter};

/// Maximum stderr tail retained in memory (16 KiB) for diagnostic reporting on
/// unexpected child exit.
#[cfg(feature = "process")]
const MAX_STDERR_TAIL_BYTES: usize = 16 * 1024;

/// Response returned by the Claude CLI for an `initialize` control request.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InitializeResponse {
    /// Slash commands advertised by the CLI.
    #[serde(default)]
    pub commands: Vec<CommandInfo>,
    /// Agent definitions available in the session.
    #[serde(default)]
    pub agents: Vec<Value>,
    /// Active output style name, if reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_style: Option<String>,
    /// Output styles selectable in this session.
    #[serde(default)]
    pub available_output_styles: Vec<String>,
    /// Selectable models advertised by the CLI.
    #[serde(default)]
    pub models: Vec<Value>,
    /// Additional forward-compatible fields returned by newer CLI releases.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Parameters for a `rewind_files` control request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RewindFilesRequest {
    /// UUID of the user message whose file checkpoint should be restored.
    pub user_message_id: String,
    /// When `Some(true)`, reports diff statistics without modifying files on disk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dry_run: Option<bool>,
}

impl RewindFilesRequest {
    /// Build a `rewind_files` request targeting `user_message_id`.
    #[must_use]
    pub fn new(user_message_id: impl Into<String>) -> Self {
        Self {
            user_message_id: user_message_id.into(),
            dry_run: None,
        }
    }

    /// Preview affected files and line counts without modifying files on disk.
    #[must_use]
    pub const fn dry_run(mut self, enabled: bool) -> Self {
        self.dry_run = Some(enabled);
        self
    }
}

/// Response returned by the Claude CLI for a `rewind_files` control request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewindFilesResponse {
    /// Whether the session can rewind (or did rewind) to the target message.
    #[serde(default)]
    pub can_rewind: bool,
    /// Diagnostic reason when `can_rewind` is `false`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Paths affected by the rewind (reported on dry runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files_changed: Option<Vec<String>>,
    /// Line insertions reverted (reported on dry runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub insertions: Option<u64>,
    /// Line deletions reverted (reported on dry runs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletions: Option<u64>,
    /// Additional fields returned by newer CLI versions.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Status of an individual MCP server reported by `mcp_status`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerStatus {
    /// Configured server name.
    #[serde(default)]
    pub name: String,
    /// Connection state (e.g. `"connected"`, `"failed"`, `"pending"`, `"needs-auth"`).
    #[serde(default)]
    pub status: String,
    /// Error message when connection failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Server implementation metadata reported on MCP initialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_info: Option<Value>,
    /// Tools exposed by this server when connected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Value>>,
    /// Additional fields preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Response returned by the Claude CLI for an `mcp_status` control request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpStatusResponse {
    /// Current status entries for configured MCP servers (`null` entries skipped).
    #[serde(
        rename = "mcpServers",
        default,
        deserialize_with = "deserialize_nullable_vec"
    )]
    pub mcp_servers: Vec<McpServerStatus>,
    /// Additional forward-compatible fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

fn deserialize_nullable_vec<'de, D, T>(deserializer: D) -> std::result::Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    let raw: Vec<Option<T>> = Vec::deserialize(deserializer)?;
    Ok(raw.into_iter().flatten().collect())
}

/// Parsed incremental delta extracted from a [`StreamEventMessage`] (`--include-partial-messages`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamDelta {
    /// Start of a new assistant message (`message_start`).
    MessageStart {
        /// Anthropic API message ID (`msg_...`), if present.
        message_id: Option<String>,
    },
    /// Start of a tool-use content block (`content_block_start` with `type: "tool_use"`).
    ToolUseStart {
        /// 0-based content block index within the message.
        index: u64,
        /// Tool invocation ID (`toolu_...`).
        id: String,
        /// Tool name (e.g. `"Bash"`, `"Edit"`).
        name: String,
    },
    /// Incremental assistant text chunk (`text_delta`).
    Text {
        /// 0-based content block index within the message.
        index: u64,
        /// Appended text fragment.
        text: String,
    },
    /// Incremental extended-thinking chunk (`thinking_delta`).
    Thinking {
        /// 0-based content block index within the message.
        index: u64,
        /// Appended thinking fragment.
        thinking: String,
    },
    /// Incremental JSON fragment for a tool call's input (`input_json_delta`).
    ToolInputJson {
        /// 0-based content block index within the message.
        index: u64,
        /// Partial JSON string slice to concatenate.
        partial_json: String,
    },
    /// Incremental thinking signature chunk (`signature_delta`).
    Signature {
        /// 0-based content block index within the message.
        index: u64,
        /// Signature fragment.
        signature: String,
    },
    /// End of a content block (`content_block_stop`).
    ContentBlockStop {
        /// 0-based content block index that finished streaming.
        index: u64,
    },
    /// Message-level stop reason update (`message_delta`).
    MessageDelta {
        /// Stop reason string (e.g. `"end_turn"`, `"tool_use"`), if present.
        stop_reason: Option<String>,
    },
    /// End of the assistant message stream (`message_stop`).
    MessageStop,
}

impl StreamDelta {
    /// Extract a typed [`StreamDelta`] from a [`StreamEventMessage`], returning
    /// `None` for events without delta semantics (such as `ping` or text/thinking
    /// `content_block_start` headers).
    #[must_use]
    pub fn from_stream_event(msg: &StreamEventMessage) -> Option<Self> {
        Self::from_raw_event(&msg.event)
    }

    /// Extract a typed [`StreamDelta`] from the inner Anthropic stream `event` JSON.
    #[must_use]
    pub fn from_raw_event(event: &Value) -> Option<Self> {
        let event_type = event.get("type")?.as_str()?;
        match event_type {
            "message_start" => {
                let message_id = event
                    .get("message")
                    .and_then(|m| m.get("id"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                Some(Self::MessageStart { message_id })
            }
            "content_block_start" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let block = event.get("content_block")?;
                if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                    let id = block.get("id")?.as_str()?.to_owned();
                    let name = block.get("name")?.as_str()?.to_owned();
                    Some(Self::ToolUseStart { index, id, name })
                } else {
                    None
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let delta = event.get("delta")?;
                match delta.get("type")?.as_str()? {
                    "text_delta" => Some(Self::Text {
                        index,
                        text: delta.get("text")?.as_str()?.to_owned(),
                    }),
                    "thinking_delta" => Some(Self::Thinking {
                        index,
                        thinking: delta.get("thinking")?.as_str()?.to_owned(),
                    }),
                    "input_json_delta" => Some(Self::ToolInputJson {
                        index,
                        partial_json: delta.get("partial_json")?.as_str()?.to_owned(),
                    }),
                    "signature_delta" => Some(Self::Signature {
                        index,
                        signature: delta.get("signature")?.as_str()?.to_owned(),
                    }),
                    _ => None,
                }
            }
            "content_block_stop" => Some(Self::ContentBlockStop {
                index: event.get("index").and_then(Value::as_u64).unwrap_or(0),
            }),
            "message_delta" => {
                let stop_reason = event
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                Some(Self::MessageDelta { stop_reason })
            }
            "message_stop" => Some(Self::MessageStop),
            _ => None,
        }
    }
}

/// Inbound session events yielded by [`Client::next_event`] and [`Client::events`].
///
/// Control-protocol frames (`control_request`, `control_response`, and
/// `control_cancel_request`) are handled directly by the session driver and are
/// never emitted here.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// System lifecycle or metadata event (`init`, `status`, `compact_boundary`, ...).
    System(SystemMessage),
    /// Incremental streaming API event emitted with `--include-partial-messages`.
    StreamEvent(StreamEventMessage),
    /// Complete or partial assistant message frame.
    Assistant(AssistantMessage),
    /// Echoed user or tool-result message frame.
    User(UserMessage),
    /// Rate-limit window utilization update.
    RateLimit(RateLimitEvent),
    /// Terminal turn result carrying stop subtype, duration, and usage totals.
    Result(ResultMessage),
    /// Anthropic API-level error frame (`500`, `529 overloaded`, ...).
    ApiError(AnthropicError),
    /// Progress heartbeat or subagent retry notice for a running tool.
    ToolProgress(ToolProgressMessage),
    /// Authentication status update.
    AuthStatus(AuthStatusMessage),
    /// Queued command lifecycle transition.
    CommandLifecycle(CommandLifecycleMessage),
    /// Summary of preceding tool invocations.
    ToolUseSummary(ToolUseSummaryMessage),
    /// Predicted follow-up prompt suggestion.
    PromptSuggestion(PromptSuggestionMessage),
    /// Mid-session conversation reset boundary (`/clear`, ...).
    ConversationReset(ConversationResetMessage),
    /// Any other non-control `ClaudeOutput` frame (e.g. transcript metadata).
    Other(ClaudeOutput),
}

impl SessionEvent {
    /// Extract a typed [`StreamDelta`] when this event is [`SessionEvent::StreamEvent`].
    #[must_use]
    pub fn as_stream_delta(&self) -> Option<StreamDelta> {
        match self {
            Self::StreamEvent(msg) => StreamDelta::from_stream_event(msg),
            _ => None,
        }
    }

    fn from_claude_output(output: ClaudeOutput) -> Self {
        match output {
            ClaudeOutput::System(msg) => Self::System(msg),
            ClaudeOutput::StreamEvent(msg) => Self::StreamEvent(msg),
            ClaudeOutput::Assistant(msg) => Self::Assistant(msg),
            ClaudeOutput::User(msg) => Self::User(msg),
            ClaudeOutput::RateLimitEvent(msg) => Self::RateLimit(msg),
            ClaudeOutput::Result(msg) => Self::Result(msg),
            ClaudeOutput::Error(err) => Self::ApiError(err),
            ClaudeOutput::ToolProgress(msg) => Self::ToolProgress(msg),
            ClaudeOutput::AuthStatus(msg) => Self::AuthStatus(msg),
            ClaudeOutput::CommandLifecycle(msg) => Self::CommandLifecycle(msg),
            ClaudeOutput::ToolUseSummary(msg) => Self::ToolUseSummary(msg),
            ClaudeOutput::PromptSuggestion(msg) => Self::PromptSuggestion(msg),
            ClaudeOutput::ConversationReset(msg) => Self::ConversationReset(msg),
            other => Self::Other(other),
        }
    }
}

#[derive(Serialize)]
struct OutboundControlEnvelope<'a> {
    #[serde(rename = "type")]
    message_type: &'static str,
    request_id: &'a str,
    request: &'a Value,
}

enum Outbound {
    ControlRequest {
        request_id: String,
        request: Value,
        reply: Sender<Result<Option<Value>>>,
    },
    Input {
        input: Box<ClaudeInput>,
        ack: Sender<Result<()>>,
    },
    Close,
}

/// Cloneable handle to an active Claude Code `stream-json` control-protocol session.
///
/// Created via [`Client::attach`] (over arbitrary async streams) or
/// [`Client::spawn`] (spawning a `claude` child process when the `process`
/// feature is enabled). All I/O runs on the companion driver future returned at
/// construction time; poll or spawn that future on any async executor.
#[derive(Clone)]
pub struct Client {
    outbound: Sender<Outbound>,
    events: Receiver<SessionEvent>,
    next_id: Arc<AtomicU64>,
    terminal_error: Arc<OnceLock<SdkError>>,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("closed", &self.outbound.is_closed())
            .finish_non_exhaustive()
    }
}

impl Client {
    /// Attach a [`Client`] to an existing `AsyncRead + AsyncWrite` byte stream
    /// pair (e.g. pipes or an in-memory duplex test transport).
    ///
    /// Returns `(client, driver)`. The caller must poll or spawn `driver` so
    /// frames are read, dispatched, and written.
    pub fn attach<R, W, H>(
        read: R,
        write: W,
        handler: H,
    ) -> (Self, impl Future<Output = Result<()>> + Send)
    where
        R: AsyncRead + Unpin + Send,
        W: AsyncWrite + Unpin + Send,
        H: ControlHandler,
    {
        let (client, state) = Self::new_pair();
        let driver = async move {
            let outcome = drive_streams(
                NdjsonReader::new(read),
                NdjsonWriter::new(write),
                Arc::new(handler),
                &state.outbound_rx,
                &state.events_tx,
            )
            .await;

            if let Some(err) = outcome.error {
                let _ = state.terminal_error.set(err.clone());
                state.outbound_rx.close();
                fail_pending(
                    outcome.pending,
                    outcome.pending_acks,
                    &state.outbound_rx,
                    &err,
                );
                state.events_tx.close();
                Err(err)
            } else {
                state.outbound_rx.close();
                fail_pending(
                    outcome.pending,
                    outcome.pending_acks,
                    &state.outbound_rx,
                    &SdkError::ChannelClosed,
                );
                state.events_tx.close();
                Ok(())
            }
        };
        (client, driver)
    }

    /// Spawn a `claude` child process from `config` and attach a [`Client`] to
    /// its piped `stdin`/`stdout`/`stderr`.
    ///
    /// Returns `(client, driver)`. When the process exits with a non-zero status,
    /// `driver` resolves to [`SdkError::ProcessExited`] (including captured
    /// `stderr`), and all pending control requests and [`Client::next_event`]
    /// calls fail immediately with that same error.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Json`] if `config` fails to serialize `--mcp-config`,
    /// or [`SdkError::Io`] if spawning the child process fails.
    #[cfg(feature = "process")]
    pub fn spawn<H>(
        config: SessionConfig,
        handler: H,
    ) -> Result<(Self, impl Future<Output = Result<ExitStatus>> + Send)>
    where
        H: ControlHandler,
    {
        let program = config.resolve_executable();
        let args = config.build_args()?;

        debug!(program = %program.display(), ?args, "spawning Claude CLI process");

        let mut cmd = async_process::Command::new(&program);
        cmd.args(&args)
            .stdin(async_process::Stdio::piped())
            .stdout(async_process::Stdio::piped())
            .stderr(async_process::Stdio::piped())
            .kill_on_drop(true);

        if let Some(cwd) = config.cwd {
            cmd.current_dir(cwd);
        }

        // Clear inherited ambient marker before applying caller-supplied env so
        // nested invocations work unless explicitly overridden in `config.env`.
        cmd.env_remove("CLAUDECODE");
        cmd.envs(config.env);

        let mut child = cmd.spawn().map_err(SdkError::from)?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| SdkError::Protocol("spawned child missing piped stdin".to_string()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| SdkError::Protocol("spawned child missing piped stdout".to_string()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| SdkError::Protocol("spawned child missing piped stderr".to_string()))?;

        let (client, state) = Self::new_pair();
        let driver = async move {
            let io_fut = drive_streams(
                NdjsonReader::new(stdout),
                NdjsonWriter::new(stdin),
                Arc::new(handler),
                &state.outbound_rx,
                &state.events_tx,
            );
            let stderr_fut = drain_stderr(stderr);

            // Run protocol I/O concurrently with stderr draining so the child
            // can never block on a full stderr pipe. Because `drive_streams`
            // drops `stdin` on exit, well-behaved CLI runs exit promptly; if
            // the client explicitly aborted or hit a protocol error, terminate
            // the child before awaiting exit status.
            let (outcome, stderr_tail) = run_child_io(&mut child, io_fut, stderr_fut).await;
            let exit_status = child.status().await.ok();

            let final_error = match (outcome.error, exit_status) {
                (err_opt, Some(status)) if !status.success() && !outcome.client_closed => {
                    if outcome.error_is_protocol {
                        err_opt
                    } else {
                        Some(SdkError::ProcessExited {
                            status: Some(status),
                            stderr: stderr_tail.clone(),
                        })
                    }
                }
                (Some(err), _) => Some(err),
                (None, None) if !outcome.client_closed => Some(SdkError::ProcessExited {
                    status: None,
                    stderr: stderr_tail.clone(),
                }),
                (None, _) => None,
            };

            if let Some(err) = final_error {
                let _ = state.terminal_error.set(err.clone());
                state.outbound_rx.close();
                fail_pending(
                    outcome.pending,
                    outcome.pending_acks,
                    &state.outbound_rx,
                    &err,
                );
                state.events_tx.close();
                Err(err)
            } else {
                state.outbound_rx.close();
                fail_pending(
                    outcome.pending,
                    outcome.pending_acks,
                    &state.outbound_rx,
                    &SdkError::ChannelClosed,
                );
                state.events_tx.close();
                exit_status.ok_or(SdkError::ProcessExited {
                    status: None,
                    stderr: stderr_tail,
                })
            }
        };

        Ok((client, driver))
    }

    fn new_pair() -> (Self, DriverChannels) {
        let (outbound_tx, outbound_rx) = async_channel::unbounded();
        let (events_tx, events_rx) = async_channel::unbounded();
        let terminal_error = Arc::new(OnceLock::new());
        let client = Self {
            outbound: outbound_tx,
            events: events_rx,
            next_id: Arc::new(AtomicU64::new(1)),
            terminal_error: Arc::clone(&terminal_error),
        };
        (
            client,
            DriverChannels {
                outbound_rx,
                events_tx,
                terminal_error,
            },
        )
    }

    /// Subscribe to the inbound [`SessionEvent`] receiver channel.
    #[must_use]
    pub fn events(&self) -> Receiver<SessionEvent> {
        self.events.clone()
    }

    /// Return the terminal error recorded when the transport or child process
    /// failed, if any.
    #[must_use]
    pub fn terminal_error(&self) -> Option<SdkError> {
        self.terminal_error.get().cloned()
    }

    /// Receive the next inbound [`SessionEvent`], returning `Ok(None)` on clean
    /// session shutdown or `Err(SdkError)` if the session terminated with an
    /// I/O, protocol, or child-exit error.
    ///
    /// # Errors
    ///
    /// Returns the terminal [`SdkError`] if the driver closed abnormally after
    /// all buffered events have been drained.
    pub async fn next_event(&self) -> Result<Option<SessionEvent>> {
        self.events.recv().await.map_or_else(
            |_| self.terminal_error().map_or(Ok(None), Err),
            |event| Ok(Some(event)),
        )
    }

    /// Request immediate shutdown of the session transport (and child process,
    /// when spawned).
    pub fn close(&self) {
        let _ = self.outbound.try_send(Outbound::Close);
        self.outbound.close();
    }

    /// Send a plain-text user prompt associated with the active CLI session.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if writing the frame fails or the session is closed.
    pub async fn send_user_message(&self, text: impl Into<String>) -> Result<()> {
        self.send_input(ClaudeInput::user_message_without_session(text))
            .await
    }

    /// Send a plain-text user prompt stamped with an explicit `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if writing the frame fails or the session is closed.
    pub async fn send_user_message_with_session(
        &self,
        text: impl Into<String>,
        session_id: Uuid,
    ) -> Result<()> {
        self.send_input(ClaudeInput::user_message(text, session_id))
            .await
    }

    /// Send a multi-block user message (text, images, tool results) with an
    /// optional `session_id`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if writing the frame fails or the session is closed.
    pub async fn send_user_blocks(
        &self,
        blocks: Vec<ContentBlock>,
        session_id: Option<Uuid>,
    ) -> Result<()> {
        let input = if let Some(id) = session_id {
            ClaudeInput::user_message_blocks(blocks, id)
        } else {
            let ClaudeInput::User(mut msg) =
                ClaudeInput::user_message_without_session(String::new())
            else {
                unreachable!()
            };
            msg.message.role = MessageRole::User;
            msg.message.content = blocks;
            ClaudeInput::User(msg)
        };
        self.send_input(input).await
    }

    /// Write an arbitrary [`ClaudeInput`] frame to the CLI's `stdin`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if serialization/writing fails or the session is closed.
    pub async fn send_input(&self, input: ClaudeInput) -> Result<()> {
        let (ack_tx, ack_rx) = async_channel::bounded(1);
        self.outbound
            .send(Outbound::Input {
                input: Box::new(input),
                ack: ack_tx,
            })
            .await
            .map_err(|_| self.closed_error())?;
        ack_rx.recv().await.map_err(|_| self.closed_error())?
    }

    /// Perform the `initialize` control-protocol handshake with default options,
    /// enabling bidirectional `can_use_tool` permission prompts.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI rejects initialization or the connection closes.
    pub async fn initialize(&self) -> Result<InitializeResponse> {
        self.initialize_with(InitializeRequest { hooks: None })
            .await
    }

    /// Perform the `initialize` control-protocol handshake with explicit
    /// [`InitializeRequest`] parameters (such as hook registrations).
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if serialization fails, the CLI returns a control
    /// error, or the connection closes.
    pub async fn initialize_with(&self, req: InitializeRequest) -> Result<InitializeResponse> {
        let payload = ControlRequestPayload::Initialize(req);
        let value =
            serde_json::to_value(payload).map_err(|err| SdkError::json("initialize", err))?;
        let resp = self.send_control("init", value).await?;
        decode_optional_response("initialize", resp)
    }

    /// Interrupt the currently running turn (`subtype: "interrupt"`), returning
    /// once the CLI acknowledges the `control_request`.
    ///
    /// Safe to invoke concurrently while a `can_use_tool` permission prompt or
    /// streaming response is in flight.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI returns a control error or the connection closes.
    pub async fn interrupt(&self) -> Result<()> {
        let value = serde_json::to_value(ControlRequestPayload::Interrupt)
            .map_err(|err| SdkError::json("interrupt", err))?;
        let _ = self.send_control("interrupt", value).await?;
        Ok(())
    }

    /// Dynamically switch the session's [`PermissionMode`] (`subtype: "set_permission_mode"`).
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI rejects the mode or the connection closes.
    pub async fn set_permission_mode(&self, mode: PermissionMode) -> Result<()> {
        let request = json!({
            "subtype": "set_permission_mode",
            "mode": mode,
        });
        let _ = self.send_control("set-mode", request).await?;
        Ok(())
    }

    /// Dynamically switch the session's active model (`subtype: "set_model"`).
    ///
    /// Pass `None` to reset to the session's default model.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI rejects the model or the connection closes.
    pub async fn set_model(&self, model: Option<&str>) -> Result<()> {
        let request = json!({
            "subtype": "set_model",
            "model": model,
        });
        let _ = self.send_control("set-model", request).await?;
        Ok(())
    }

    /// Dynamically set or clear the extended-thinking token ceiling
    /// (`subtype: "set_max_thinking_tokens"`).
    ///
    /// Pass `Some(0)` to disable extended thinking or `None` to reset to the
    /// session default.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI rejects the value or the connection closes.
    pub async fn set_max_thinking_tokens(&self, max_thinking_tokens: Option<u32>) -> Result<()> {
        let request = json!({
            "subtype": "set_max_thinking_tokens",
            "max_thinking_tokens": max_thinking_tokens,
        });
        let _ = self.send_control("set-thinking", request).await?;
        Ok(())
    }

    /// Merge flag settings into the running session (`subtype: "apply_flag_settings"`).
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if serialization fails, the CLI rejects a setting,
    /// or the connection closes.
    pub async fn apply_flag_settings(&self, settings: ApplyFlagSettingsRequest) -> Result<()> {
        let value = serde_json::to_value(ControlRequestPayload::ApplyFlagSettings(settings))
            .map_err(|err| SdkError::json("apply_flag_settings", err))?;
        let _ = self.send_control("apply-flag-settings", value).await?;
        Ok(())
    }

    /// Change the session's [`EffortLevel`] mid-session via `apply_flag_settings`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI rejects the level or the connection closes.
    pub async fn set_effort(&self, effort: EffortLevel) -> Result<()> {
        self.apply_flag_settings(ApplyFlagSettingsRequest::new().effort_level(effort))
            .await
    }

    /// Rewind workspace files modified since `user_message_id`
    /// (`subtype: "rewind_files"`).
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI reports a rewind error or the connection closes.
    pub async fn rewind_files(
        &self,
        user_message_id: impl Into<String>,
    ) -> Result<RewindFilesResponse> {
        self.rewind_files_with_options(RewindFilesRequest::new(user_message_id))
            .await
    }

    /// Rewind workspace files (or preview diff stats with `dry_run: Some(true)`)
    /// using [`RewindFilesRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if serialization fails, the CLI rejects the rewind,
    /// or the connection closes.
    pub async fn rewind_files_with_options(
        &self,
        req: RewindFilesRequest,
    ) -> Result<RewindFilesResponse> {
        let mut request =
            serde_json::to_value(&req).map_err(|err| SdkError::json("rewind_files", err))?;
        if let Some(obj) = request.as_object_mut() {
            obj.insert(
                "subtype".to_string(),
                Value::String("rewind_files".to_string()),
            );
        }
        let resp = self.send_control("rewind-files", request).await?;
        resp.map_or_else(
            || {
                Ok(RewindFilesResponse {
                    can_rewind: true,
                    ..RewindFilesResponse::default()
                })
            },
            |val| serde_json::from_value(val).map_err(|err| SdkError::json("rewind_files", err)),
        )
    }

    /// Query connection and tool status for all configured MCP servers
    /// (`subtype: "mcp_status"`).
    ///
    /// # Errors
    ///
    /// Returns [`SdkError`] if the CLI returns a control error or the connection closes.
    pub async fn mcp_status(&self) -> Result<McpStatusResponse> {
        let request = json!({ "subtype": "mcp_status" });
        let resp = self.send_control("mcp-status", request).await?;
        decode_optional_response("mcp_status", resp)
    }

    /// Send an arbitrary outbound `control_request` body (must be a JSON object
    /// containing a string `"subtype"` field) and await its correlated response.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Protocol`] if `request` lacks a string `"subtype"`
    /// field, [`SdkError::Control`] if the CLI responds with an error, or
    /// [`SdkError::ChannelClosed`] / [`SdkError::ProcessExited`] if the session ends.
    pub async fn control_request(&self, request: Value) -> Result<Option<Value>> {
        let Some(subtype) = request
            .as_object()
            .and_then(|obj| obj.get("subtype"))
            .and_then(Value::as_str)
        else {
            return Err(SdkError::Protocol(
                "control_request payload must be a JSON object with a string `subtype`".to_string(),
            ));
        };
        let prefix = subtype.replace('_', "-");
        self.send_control(&prefix, request).await
    }

    async fn send_control(&self, prefix: &str, request: Value) -> Result<Option<Value>> {
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request_id = format!("{prefix}-{seq}-{}", Uuid::new_v4());
        let (reply_tx, reply_rx) = async_channel::bounded(1);

        self.outbound
            .send(Outbound::ControlRequest {
                request_id,
                request,
                reply: reply_tx,
            })
            .await
            .map_err(|_| self.closed_error())?;

        reply_rx.recv().await.map_err(|_| self.closed_error())?
    }

    fn closed_error(&self) -> SdkError {
        self.terminal_error().unwrap_or(SdkError::ChannelClosed)
    }
}

fn decode_optional_response<T>(context: &'static str, resp: Option<Value>) -> Result<T>
where
    T: DeserializeOwned + Default,
{
    match resp {
        Some(Value::Null) | None => Ok(T::default()),
        Some(value) => serde_json::from_value(value).map_err(|err| SdkError::json(context, err)),
    }
}

struct DriverChannels {
    outbound_rx: Receiver<Outbound>,
    events_tx: Sender<SessionEvent>,
    terminal_error: Arc<OnceLock<SdkError>>,
}

struct DriveOutcome {
    pending: HashMap<String, Sender<Result<Option<Value>>>>,
    pending_acks: Vec<Sender<Result<()>>>,
    error: Option<SdkError>,
    #[cfg_attr(not(feature = "process"), allow(dead_code))]
    error_is_protocol: bool,
    #[cfg_attr(not(feature = "process"), allow(dead_code))]
    client_closed: bool,
}

impl DriveOutcome {
    const fn ok(
        pending: HashMap<String, Sender<Result<Option<Value>>>>,
        pending_acks: Vec<Sender<Result<()>>>,
        client_closed: bool,
    ) -> Self {
        Self {
            pending,
            pending_acks,
            error: None,
            error_is_protocol: false,
            client_closed,
        }
    }

    const fn err(
        pending: HashMap<String, Sender<Result<Option<Value>>>>,
        pending_acks: Vec<Sender<Result<()>>>,
        error: SdkError,
        error_is_protocol: bool,
    ) -> Self {
        Self {
            pending,
            pending_acks,
            error: Some(error),
            error_is_protocol,
            client_closed: false,
        }
    }
}

struct InFlightTask {
    request_id: Option<String>,
    future: Pin<Box<dyn Future<Output = Option<ControlResponseMessage>> + Send>>,
}

enum Step {
    HandlerReply(ControlResponseMessage),
    Outbound(std::result::Result<Outbound, async_channel::RecvError>),
    Inbound(Result<Option<Box<ClaudeOutput>>>),
}

async fn next_in_flight(tasks: &mut Vec<InFlightTask>) -> ControlResponseMessage {
    poll_fn(|cx| {
        let mut idx = 0;
        while idx < tasks.len() {
            if let Poll::Ready(result) = tasks[idx].future.as_mut().poll(cx) {
                drop(tasks.swap_remove(idx));
                if let Some(message) = result {
                    return Poll::Ready(message);
                }
                continue;
            }
            idx += 1;
        }
        Poll::Pending
    })
    .await
}

async fn next_outbound(
    outbound_rx: &Receiver<Outbound>,
    outbound_closed: bool,
) -> std::result::Result<Outbound, async_channel::RecvError> {
    if outbound_closed {
        std::future::pending().await
    } else {
        outbound_rx.recv().await
    }
}

async fn drive_streams<R, W, H>(
    mut reader: NdjsonReader<R>,
    mut writer: NdjsonWriter<W>,
    handler: Arc<H>,
    outbound_rx: &Receiver<Outbound>,
    events_tx: &Sender<SessionEvent>,
) -> DriveOutcome
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
    H: ControlHandler,
{
    let mut pending: HashMap<String, Sender<Result<Option<Value>>>> = HashMap::new();
    let mut pending_acks: Vec<Sender<Result<()>>> = Vec::new();
    let mut in_flight: Vec<InFlightTask> = Vec::new();
    let mut outbound_closed = false;
    let mut stdin_closed = false;

    loop {
        if outbound_closed && !stdin_closed && in_flight.is_empty() && pending.is_empty() {
            let _ = writer.close().await;
            stdin_closed = true;
        }

        let step = or(
            or(
                async { Step::HandlerReply(next_in_flight(&mut in_flight).await) },
                async { Step::Outbound(next_outbound(outbound_rx, outbound_closed).await) },
            ),
            async { Step::Inbound(reader.recv_output().await.map(|opt| opt.map(Box::new))) },
        )
        .await;

        match step {
            Step::HandlerReply(response_msg) => {
                if let Err(err) = writer.send_json(&response_msg).await {
                    let is_protocol = matches!(err, SdkError::Protocol(_) | SdkError::Json { .. });
                    let _ = writer.close().await;
                    return DriveOutcome::err(pending, pending_acks, err, is_protocol);
                }
            }
            Step::Outbound(Ok(Outbound::ControlRequest {
                request_id,
                request,
                reply,
            })) => {
                let envelope = OutboundControlEnvelope {
                    message_type: "control_request",
                    request_id: &request_id,
                    request: &request,
                };
                let send_res = writer.send_json(&envelope).await;
                pending.insert(request_id, reply);
                if let Err(err) = send_res {
                    let is_protocol = matches!(err, SdkError::Protocol(_) | SdkError::Json { .. });
                    let _ = writer.close().await;
                    return DriveOutcome::err(pending, pending_acks, err, is_protocol);
                }
            }
            Step::Outbound(Ok(Outbound::Input { input, ack })) => {
                if let Err(err) = writer.send_json(&input).await {
                    let is_protocol = matches!(err, SdkError::Protocol(_) | SdkError::Json { .. });
                    pending_acks.push(ack);
                    let _ = writer.close().await;
                    return DriveOutcome::err(pending, pending_acks, err, is_protocol);
                }
                let _ = ack.try_send(Ok(()));
            }
            Step::Outbound(Ok(Outbound::Close)) => {
                let _ = writer.close().await;
                return DriveOutcome::ok(pending, pending_acks, true);
            }
            Step::Outbound(Err(_)) => {
                outbound_closed = true;
            }
            Step::Inbound(Ok(None)) => {
                let _ = writer.close().await;
                return DriveOutcome::ok(pending, pending_acks, false);
            }
            Step::Inbound(Ok(Some(output))) => {
                handle_inbound_output(*output, &handler, &mut pending, &mut in_flight, events_tx);
            }
            Step::Inbound(Err(err)) => {
                let is_protocol = matches!(err, SdkError::Protocol(_) | SdkError::Json { .. });
                let _ = writer.close().await;
                return DriveOutcome::err(pending, pending_acks, err, is_protocol);
            }
        }
    }
}

fn handle_inbound_output<H: ControlHandler>(
    output: ClaudeOutput,
    handler: &Arc<H>,
    pending: &mut HashMap<String, Sender<Result<Option<Value>>>>,
    in_flight: &mut Vec<InFlightTask>,
    events_tx: &Sender<SessionEvent>,
) {
    match output {
        ClaudeOutput::ControlResponse(resp) => match resp.response {
            ControlResponsePayload::Success {
                request_id,
                response,
            } => {
                if let Some(waiter) = pending.remove(&request_id) {
                    let _ = waiter.try_send(Ok(response));
                } else {
                    debug!(%request_id, "ignoring control_response for unknown request_id");
                }
            }
            ControlResponsePayload::Error { request_id, error } => {
                if let Some(waiter) = pending.remove(&request_id) {
                    let _ = waiter.try_send(Err(SdkError::Control {
                        request_id,
                        message: error,
                    }));
                } else {
                    warn!(%request_id, %error, "received unmatched control_response error");
                }
            }
        },
        ClaudeOutput::ControlRequest(req) => {
            let req_id = req.request_id.clone();
            let handler = Arc::clone(handler);
            in_flight.push(InFlightTask {
                request_id: Some(req_id),
                future: Box::pin(async move { Some(dispatch_inbound_control(handler, req).await) }),
            });
        }
        ClaudeOutput::ControlCancelRequest(cancel) => {
            in_flight.retain(|task| task.request_id.as_deref() != Some(cancel.request_id.as_str()));
            let handler = Arc::clone(handler);
            in_flight.push(InFlightTask {
                request_id: None,
                future: Box::pin(async move {
                    handler.cancel_request(cancel).await;
                    None
                }),
            });
        }
        other => {
            let _ = events_tx.try_send(SessionEvent::from_claude_output(other));
        }
    }
}

async fn dispatch_inbound_control<H: ControlHandler>(
    handler: Arc<H>,
    req: ControlRequest,
) -> ControlResponseMessage {
    let request_id = req.request_id;
    let response = match req.request {
        ControlRequestPayload::CanUseTool(perm_req) => {
            let original_input = perm_req.input.clone();
            match handler.can_use_tool(perm_req).await {
                Ok(decision) => match decision.into_permission_result(original_input) {
                    Ok(result) => ControlResponse::from_result(&request_id, result),
                    Err(err) => ControlResponse::error(&request_id, err.to_string()),
                },
                Err(err) => ControlResponse::error(&request_id, err.to_string()),
            }
        }
        ControlRequestPayload::HookCallback(hook_req) => {
            match handler.hook_callback(hook_req).await {
                Ok(payload) => ControlResponse::success(&request_id, payload),
                Err(err) => ControlResponse::error(&request_id, err.to_string()),
            }
        }
        ControlRequestPayload::McpMessage(mcp_req) => match handler.mcp_message(mcp_req).await {
            Ok(mcp_reply) => {
                ControlResponse::success(&request_id, json!({ "mcp_response": mcp_reply }))
            }
            Err(err) => ControlResponse::error(&request_id, err.to_string()),
        },
        other => ControlResponse::error(
            &request_id,
            format!("unsupported inbound control_request subtype: {other:?}"),
        ),
    };
    ControlResponseMessage::from(response)
}

fn fail_pending(
    pending: HashMap<String, Sender<Result<Option<Value>>>>,
    pending_acks: Vec<Sender<Result<()>>>,
    outbound_rx: &Receiver<Outbound>,
    err: &SdkError,
) {
    for (_, waiter) in pending {
        let _ = waiter.try_send(Err(err.clone()));
    }
    for ack in pending_acks {
        let _ = ack.try_send(Err(err.clone()));
    }
    while let Ok(item) = outbound_rx.try_recv() {
        match item {
            Outbound::ControlRequest { reply, .. } => {
                let _ = reply.try_send(Err(err.clone()));
            }
            Outbound::Input { ack, .. } => {
                let _ = ack.try_send(Err(err.clone()));
            }
            Outbound::Close => {}
        }
    }
}

#[cfg(feature = "process")]
async fn run_child_io<IoFut, ErrFut>(
    child: &mut async_process::Child,
    io_fut: IoFut,
    stderr_fut: ErrFut,
) -> (DriveOutcome, String)
where
    IoFut: Future<Output = DriveOutcome>,
    ErrFut: Future<Output = String>,
{
    enum FirstDone {
        Io(DriveOutcome),
        Stderr(String),
    }

    let mut io_pinned = Box::pin(io_fut);
    let mut stderr_pinned = Box::pin(stderr_fut);
    match or(async { FirstDone::Io((&mut io_pinned).await) }, async {
        FirstDone::Stderr((&mut stderr_pinned).await)
    })
    .await
    {
        FirstDone::Io(outcome) => {
            if outcome.client_closed {
                let _ = child.kill();
                return (outcome, String::new());
            }
            if outcome.error.is_some() {
                let _ = child.kill();
                if outcome.error_is_protocol {
                    return (outcome, String::new());
                }
            }
            drop(io_pinned);
            let stderr_tail = stderr_pinned.await;
            (outcome, stderr_tail)
        }
        FirstDone::Stderr(stderr_tail) => {
            drop(stderr_pinned);
            let outcome = io_pinned.await;
            if outcome.client_closed || outcome.error.is_some() {
                let _ = child.kill();
            }
            (outcome, stderr_tail)
        }
    }
}

#[cfg(feature = "process")]
async fn drain_stderr(stderr: async_process::ChildStderr) -> String {
    let mut reader = BufReader::new(stderr);
    let mut raw = Vec::new();
    let mut chunk = [0_u8; 4096];
    loop {
        match reader.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if let Ok(text) = std::str::from_utf8(&chunk[..n]) {
                    for line in text.lines() {
                        if !line.trim().is_empty() {
                            debug!(target: "claude_acp_sdk::stderr", "{line}");
                        }
                    }
                }
                raw.extend_from_slice(&chunk[..n]);
                if raw.len() > MAX_STDERR_TAIL_BYTES * 2 {
                    let drain_to = raw.len() - MAX_STDERR_TAIL_BYTES;
                    raw.drain(..drain_to);
                }
            }
        }
    }
    if raw.len() > MAX_STDERR_TAIL_BYTES {
        let start = raw.len() - MAX_STDERR_TAIL_BYTES;
        String::from_utf8_lossy(&raw[start..]).into_owned()
    } else {
        String::from_utf8_lossy(&raw).into_owned()
    }
}
