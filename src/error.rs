//! Error types for `claude-acp` and conversions to JSON-RPC errors.

use aither_acp::{JsonRpcError, ServerError};
use aither_mcp::protocol::ErrorCode;
use claude_acp_sdk::SdkError;

/// Convenience result alias returning [`ClaudeAcpError`].
pub type Result<T> = std::result::Result<T, ClaudeAcpError>;

/// Errors produced by the `claude-acp` agent bridge.
#[derive(Debug, thiserror::Error)]
pub enum ClaudeAcpError {
    /// Underlying `claude-acp-sdk` error.
    #[error(transparent)]
    Sdk(#[from] SdkError),

    /// Underlying `aither-acp` server transport/peer error.
    #[error(transparent)]
    Server(#[from] ServerError),

    /// Referenced session does not exist in memory or on disk.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Invalid request parameter supplied by the ACP client.
    #[error("{0}")]
    InvalidParams(String),

    /// Authentication failed or credentials are missing.
    #[error("{0}")]
    Auth(String),
}

impl ClaudeAcpError {
    /// Convert this error into a protocol [`JsonRpcError`].
    #[must_use]
    pub fn into_rpc_error(self) -> JsonRpcError {
        match self {
            Self::SessionNotFound(id) => {
                JsonRpcError::invalid_params(format!("Session not found: {id}"))
            }
            Self::InvalidParams(msg) => JsonRpcError::invalid_params(msg),
            Self::Auth(msg) => JsonRpcError::new(ErrorCode(-32000), msg),
            Self::Sdk(SdkError::Control { message, .. }) => JsonRpcError::invalid_params(message),
            Self::Sdk(err) => JsonRpcError::internal_error(err.to_string()),
            Self::Server(ServerError::JsonRpc(rpc)) => rpc,
            Self::Server(err) => JsonRpcError::internal_error(err.to_string()),
        }
    }
}

impl From<ClaudeAcpError> for JsonRpcError {
    fn from(err: ClaudeAcpError) -> Self {
        err.into_rpc_error()
    }
}
