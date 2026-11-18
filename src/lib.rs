//! Agent Client Protocol (ACP v1) server backed by Claude Code (`claude-acp-sdk`)
//! and built onto [`aither_acp::AgentHandler`] and [`ClientPeer`] abstractions.
//!
//! # Architecture
//!
//! `claude-acp` bridges editor/orchestrator ACP JSON-RPC traffic onto Claude
//! Code's bidirectional `stream-json` control protocol:
//!
//! - [`ClaudeAgent`] implements [`aither_acp::AgentHandler`] without tying library
//!   code to any specific async runtime.
//! - [`serve`] drives a [`ClaudeAgent`] over any [`BidirectionalTransport`]
//!   (such as pipes, sockets, or [`aither_mcp::transport::DuplexTransport`]),
//!   cleaning up all live sessions when the connection closes.
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
use aither_acp::ClientPeer;
use aither_mcp::transport::{BidirectionalTransport, StdioTransport};
pub use claude_acp_sdk as sdk;

pub use agent::ClaudeAgent;
pub use auth::{AUTH_METHOD_API_KEY, AUTH_METHOD_CLAUDE_LOGIN};
#[cfg(feature = "process")]
pub use config::ProcessSessionConnector;
pub use config::{ClaudeAcpConfig, ConnectedSession, SessionConnector};
pub use error::{ClaudeAcpError, Result};
pub use mode::AcpMode;

/// Serve a [`ClaudeAgent`] configured with `config` over an arbitrary
/// [`BidirectionalTransport`].
///
/// Returns a [`ClientPeer`] handle for agent-initiated notifications/requests
/// alongside the connection future that drives the transport and shuts down all
/// live SDK sessions when the connection ends.
pub fn serve<T>(
    transport: T,
    config: ClaudeAcpConfig,
) -> (ClientPeer, impl Future<Output = ()> + Send)
where
    T: BidirectionalTransport,
{
    let agent = ClaudeAgent::new(config);
    let shutdown_agent = agent.clone();
    let (peer, connection) = aither_acp::serve(transport, agent);
    let wrapped = async move {
        connection.await;
        shutdown_agent.shutdown_all().await;
    };
    (peer, wrapped)
}

/// Serve a [`ClaudeAgent`] configured with `config` over standard input and output.
pub fn serve_stdio(config: ClaudeAcpConfig) -> (ClientPeer, impl Future<Output = ()> + Send) {
    serve(StdioTransport::new(), config)
}
