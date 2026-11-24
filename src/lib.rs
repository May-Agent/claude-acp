//! Agent Client Protocol (ACP v1+v2) server backed by Claude Code
//! (`claude-acp-sdk`) and built onto the rewritten [`aither_acp`] dual-stack
//! `Server`/`SessionHost`/`Session` API.
//!
//! # Architecture
//!
//! `claude-acp` bridges editor/orchestrator ACP JSON-RPC traffic onto Claude
//! Code's bidirectional `stream-json` control protocol:
//!
//! - [`ClaudeAgent`] implements [`aither_acp::server::SessionHost`] without
//!   tying library code to any specific async runtime; each ACP session is a
//!   [`ClaudeSession`] implementing [`aither_acp::server::Session`].
//! - [`serve`] drives a [`ClaudeAgent`] over any [`aither_acp::Transport`]
//!   (NDJSON lines, in-memory pairs, child processes), cleaning up all live
//!   sessions when the connection closes.
//! - [`serve_stdio`] runs the server over process `stdin`/`stdout`.
//! - [`SessionConnector`] decouples session transport creation so integration
//!   tests can attach in-memory mock CLI streams via
//!   [`ClaudeAcpConfig::connector`] while production runs spawn `claude` child
//!   processes (`process` feature, enabled by default).
//!
//! # Example
//!
//! ```no_run
//! use claude_acp::{ClaudeAcpConfig, serve_stdio};
//!
//! # async fn run() {
//! let config = ClaudeAcpConfig::new().default_mode("workspace-write");
//! let (_peer, server) = serve_stdio(config);
//! server.await;
//! # }
//! ```

mod agent;
mod auth;
mod config;
mod convert;
mod driver;
mod error;
mod mode;
mod permission;
mod session;

use std::future::Future;

pub use aither_acp as acp;
use aither_acp::rpc::Peer;
pub use claude_acp_sdk as sdk;

pub use agent::{
    ClaudeAgent, EXT_CONTEXT_USAGE, EXT_MCP_STATUS, EXT_RATE_LIMITS, NOTIFY_RATE_LIMIT,
};
pub use auth::{AUTH_METHOD_API_KEY, AUTH_METHOD_CLAUDE_LOGIN};
pub use claude_acp_sdk::{AnyTranscriptSync, NoopTranscriptSync, TranscriptSync};
#[cfg(feature = "process")]
pub use config::ProcessSessionConnector;
pub use config::{ClaudeAcpConfig, ConnectedSession, SessionConnector};
pub use error::{ClaudeAcpError, Result};
pub use mode::AcpMode;
pub use session::ClaudeSession;

/// Serve a [`ClaudeAgent`] configured with `config` over an arbitrary
/// [`aither_acp::Transport`].
///
/// Returns a [`Peer`] handle for agent-initiated notifications/requests
/// alongside the connection future that drives the transport and shuts down
/// all live SDK sessions when the connection ends.
pub fn serve<T>(transport: T, config: ClaudeAcpConfig) -> (Peer, impl Future<Output = ()> + Send)
where
    T: aither_acp::Transport + 'static,
{
    let agent = ClaudeAgent::new(config);
    let server = agent.clone().into_server();
    let (peer, connection) = server.serve(transport);
    agent.set_peer(peer.clone());
    (peer, connection)
}

/// Serve a [`ClaudeAgent`] configured with `config` over standard input and
/// output.
pub fn serve_stdio(config: ClaudeAcpConfig) -> (Peer, impl Future<Output = ()> + Send) {
    let agent = ClaudeAgent::new(config);
    let server = agent.clone().into_server();
    let (peer, connection) = server.serve_stdio();
    agent.set_peer(peer.clone());
    (peer, connection)
}
