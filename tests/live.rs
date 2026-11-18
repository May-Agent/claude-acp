//! Live end-to-end smoke test driving `claude-acp` via `aither_acp::AcpClient`
//! against a real authenticated `claude` CLI binary (`cargo test -- --ignored`).

#![cfg(feature = "process")]

use std::future::Future;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_acp::{
    AcpClient, ClientHandler, ContentBlock, PermissionOptionKind, PromptParams,
    RequestPermissionOutcome, RequestPermissionParams, RequestPermissionResult, SessionCloseParams,
    SessionNewParams, SessionNotification, SessionUpdate, StopReason, TextContent,
};
use aither_mcp::protocol::JsonRpcError;
use aither_mcp::transport::DuplexTransport;
use claude_acp::sdk::{ClaudeModel, SpawnConfig};
use claude_acp::{ClaudeAcpConfig, serve};

#[derive(Clone, Default)]
struct LiveCollector {
    chunks: Arc<Mutex<Vec<String>>>,
}

impl ClientHandler for LiveCollector {
    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send {
        if let SessionUpdate::AgentMessageChunk(chunk) = notification.update
            && let ContentBlock::Text(text) = chunk.content
        {
            self.chunks.lock().unwrap().push(text.text);
        }
        std::future::ready(())
    }

    fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        let allow_id = params
            .options
            .iter()
            .find(|opt| opt.kind == PermissionOptionKind::AllowOnce)
            .map(|opt| opt.option_id.clone());
        let outcome = allow_id.map_or(RequestPermissionOutcome::Cancelled, |option_id| {
            RequestPermissionOutcome::Selected { option_id }
        });
        std::future::ready(Ok(RequestPermissionResult {
            outcome,
            meta: None,
        }))
    }
}

#[tokio::test]
#[ignore = "requires an authenticated local `claude` CLI installation"]
async fn live_acp_round_trip_with_real_claude_cli() {
    let collector = LiveCollector::default();
    let (client_transport, server_transport) = DuplexTransport::pair();

    let spawn = SpawnConfig::new()
        .model(ClaudeModel::Haiku)
        .include_partial_messages(true)
        .extra_args(["--no-session-persistence"]);
    let config = ClaudeAcpConfig::new()
        .spawn(spawn)
        .default_mode("read-only");

    let (_peer, server_conn) = serve(server_transport, config);
    let (client, client_conn) = AcpClient::connect(client_transport, collector.clone());

    let server_task = tokio::spawn(server_conn);
    let client_task = tokio::spawn(client_conn);

    let init = tokio::time::timeout(Duration::from_secs(15), client.initialize())
        .await
        .expect("ACP initialize timed out")
        .expect("ACP initialize failed");
    assert!(init.agent_capabilities.load_session);

    let cwd = std::env::temp_dir();
    let session = tokio::time::timeout(
        Duration::from_secs(30),
        client.new_session(SessionNewParams::new(&cwd)),
    )
    .await
    .expect("session/new timed out")
    .expect("session/new failed");

    let prompt_res = tokio::time::timeout(
        Duration::from_secs(60),
        client.prompt(PromptParams::new(
            &session.session_id,
            vec![ContentBlock::Text(TextContent {
                text: "Reply with the exact word PONG and nothing else.".to_string(),
                annotations: None,
                meta: None,
            })],
        )),
    )
    .await
    .expect("session/prompt timed out")
    .expect("session/prompt failed");

    assert_eq!(prompt_res.stop_reason, StopReason::EndTurn);

    let combined = collector.chunks.lock().unwrap().join("");
    assert!(
        combined.to_uppercase().contains("PONG"),
        "expected PONG in streamed chunks, got: {combined:?}"
    );

    client
        .close_session(SessionCloseParams::new(&session.session_id))
        .await
        .expect("session/close failed");
    client.close();

    tokio::time::timeout(Duration::from_secs(10), client_task)
        .await
        .expect("client shutdown timed out")
        .expect("client task panicked");
    tokio::time::timeout(Duration::from_secs(10), server_task)
        .await
        .expect("server shutdown timed out")
        .expect("server task panicked");
}
