//! Runtime-agnostic SDK for driving Claude Code over its `stream-json`
//! bidirectional control protocol.
//!
//! `claude-acp-sdk` wraps [`claude_codes`]'s `types`-only wire models with an
//! executor-independent LF-only NDJSON transport, process launcher, concurrent
//! control-protocol router, and non-blocking session transcript helpers.
//!
//! # Example
//!
//! ```
//! use claude_acp_sdk::{
//!     ClaudeModel, PermissionDecision, PermissionMode, SessionConfig,
//! };
//!
//! let config = SessionConfig::new()
//!     .model(ClaudeModel::Sonnet)
//!     .permission_mode(PermissionMode::AcceptEdits)
//!     .include_partial_messages(true);
//!
//! let args = config.build_args().expect("valid CLI arguments");
//! assert!(args.contains(&"--permission-prompt-tool".to_string()));
//! assert!(matches!(PermissionDecision::allow(), PermissionDecision::Allow { .. }));
//! ```

pub mod client;
pub mod config;
pub mod error;
pub mod handler;
pub mod transcript;
pub mod transport;

pub use claude_codes;
pub use claude_codes::io::{
    AssistantMessage, McpHttpServerConfig, McpServerConfig, McpSseServerConfig,
    McpStdioServerConfig, ResultMessage, ResultSubtype, UserMessage,
};
pub use claude_codes::{
    AnthropicError, ApplyFlagSettingsRequest, AuthStatusMessage, ClaudeInput, ClaudeModel,
    ClaudeOutput, CommandInfo, CommandLifecycleMessage, ContentBlock, ControlCancelRequest,
    ControlRequest, ControlRequestPayload, ControlResponse, ControlResponsePayload,
    ConversationResetMessage, EffortLevel, HookCallbackRequest, ImageBlock, ImageSource,
    ImageSourceType, InitMessage, InitializeRequest, McpMessageRequest, MediaType, Permission,
    PermissionBehavior, PermissionDestination, PermissionModeName, PermissionResult,
    PermissionRule, PermissionSuggestion, PermissionType, PromptSuggestionMessage, RateLimitEvent,
    RateLimitInfo, RateLimitStatus, RateLimitWindow, StreamEventMessage, SystemMessage,
    SystemSubtype, TextBlock, ThinkingBlock, ToolInput, ToolPermissionRequest, ToolProgressMessage,
    ToolResultBlock, ToolResultContent, ToolUseBlock, ToolUseSummaryMessage,
};

pub use client::{
    Client, ContextUsageAgent, ContextUsageCategory, ContextUsageMcpTool, ContextUsageMemoryFile,
    ContextUsageResponse, ContextUsageSkill, ContextUsageSkills, InitializeResponse,
    McpServerStatus, McpStatusResponse, RewindFilesRequest, RewindFilesResponse, SessionEvent,
    StreamDelta,
};
pub use config::{PermissionMode, SessionConfig, SpawnConfig};
pub use error::{Result, SdkError};
pub use handler::{AnyControlHandler, ControlHandler, NoopControlHandler, PermissionDecision};
pub use transcript::{
    AnyTranscriptSync, NoopTranscriptSync, SessionSummary, TranscriptEntry, TranscriptRecord,
    TranscriptStore, TranscriptSync, delete_transcript, encode_project_dir, find_transcript_path,
    list_sessions, load_transcript, subagents_dir, transcript_path,
};
pub use transport::{NdjsonReader, NdjsonTransport, NdjsonWriter};
