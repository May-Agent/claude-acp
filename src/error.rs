//! Error types for `claude-acp` and conversions to JSON-RPC errors.

use aither_acp::rpc::RpcError;
use aither_acp::server::ServerError;
use claude_acp_sdk::SdkError;

/// Convenience result alias returning [`ClaudeAcpError`].
pub type Result<T> = std::result::Result<T, ClaudeAcpError>;

/// Errors produced by the `claude-acp` agent bridge.
#[derive(Debug, thiserror::Error)]
pub enum ClaudeAcpError {
    /// Underlying `claude-acp-sdk` error.
    #[error(transparent)]
    Sdk(#[from] SdkError),

    /// Underlying `aither-acp` server error.
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
    /// Convert this error into a protocol [`RpcError`].
    #[must_use]
    pub fn into_rpc_error(self) -> RpcError {
        match self {
            Self::SessionNotFound(id) => {
                RpcError::invalid_params(format!("Session not found: {id}"))
            }
            Self::InvalidParams(msg) => RpcError::invalid_params(msg),
            Self::Auth(msg) => RpcError::new(-32000, msg),
            Self::Sdk(SdkError::Control { message, .. }) => RpcError::invalid_params(message),
            Self::Sdk(err) => RpcError::internal_error(err.to_string()),
            Self::Server(ServerError::Rpc(rpc)) => rpc,
            Self::Server(err) => RpcError::internal_error(err.to_string()),
        }
    }
}

impl From<ClaudeAcpError> for RpcError {
    fn from(err: ClaudeAcpError) -> Self {
        err.into_rpc_error()
    }
}
