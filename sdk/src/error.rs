//! Error types for `claude-acp-sdk`.

use std::process::ExitStatus;
use std::sync::Arc;

/// Convenience result alias returning [`SdkError`].
pub type Result<T> = std::result::Result<T, SdkError>;

/// Errors produced by `claude-acp-sdk` transport, control protocol, process
/// management, and transcript operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SdkError {
    /// Underlying pipe, stream, or filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[source] Arc<std::io::Error>),

    /// JSON serialization or deserialization failure.
    #[error("JSON error ({context}): {source}")]
    Json {
        /// Context describing what payload failed to serialize or deserialize.
        context: &'static str,
        /// Underlying `serde_json` error.
        #[source]
        source: Arc<serde_json::Error>,
    },

    /// Wire framing or control-protocol schema violation.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Claude CLI child process exited unexpectedly.
    #[error("Claude CLI exited ({}){}", format_exit_status(*.status), format_stderr(stderr))]
    ProcessExited {
        /// Exit status reported by the operating system, if available.
        status: Option<ExitStatus>,
        /// Captured stderr output from the child process.
        stderr: String,
    },

    /// Error response returned by the Claude CLI for a `control_request`.
    #[error("control request `{request_id}` failed: {message}")]
    Control {
        /// Correlated `request_id` that failed.
        request_id: String,
        /// Error message returned by the CLI.
        message: String,
    },

    /// Session channel or transport closed before the operation completed.
    #[error("session channel closed")]
    ChannelClosed,
}

impl SdkError {
    /// Construct a [`SdkError::Json`] carrying explicit operation context.
    #[must_use]
    pub fn json(context: &'static str, source: serde_json::Error) -> Self {
        Self::Json {
            context,
            source: Arc::new(source),
        }
    }
}

impl From<std::io::Error> for SdkError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(Arc::new(error))
    }
}

impl From<serde_json::Error> for SdkError {
    fn from(error: serde_json::Error) -> Self {
        Self::json("serde_json", error)
    }
}

impl From<claude_codes::ParseError> for SdkError {
    fn from(error: claude_codes::ParseError) -> Self {
        Self::Protocol(error.error_message)
    }
}

fn format_exit_status(status: Option<ExitStatus>) -> String {
    status.map_or_else(
        || "unknown status".to_string(),
        |s| {
            s.code()
                .map_or_else(|| s.to_string(), |code| format!("code {code}"))
        },
    )
}

fn format_stderr(stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!(": {trimmed}")
    }
}
