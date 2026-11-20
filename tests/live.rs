//! Live end-to-end smoke test driving `claude-acp` via `aither_acp::Client`
//! against a real authenticated `claude` CLI binary (`cargo test -- --ignored`).

#![cfg(feature = "process")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aither_acp::domain::{
    CancelledOutcome, ContentBlock, Open, PermissionOptionKind, PermissionOutcome,
    PermissionRequest, SelectedOutcome, StopReason, TextContent,
};
use aither_acp::rpc::Memory;
use aither_acp::{Client, Implementation, TurnOutcome};
use claude_acp::sdk::{ClaudeModel, SpawnConfig};
use claude_acp::{ClaudeAcpConfig, serve};

#[tokio::test]
#[ignore = "requires an authenticated local `claude` CLI installation"]
async fn live_acp_round_trip_with_real_claude_cli() {
    let chunks: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (client_transport, server_transport) = Memory::pair();

    let spawn = SpawnConfig::new()
        .model(ClaudeModel::Haiku)
        .include_partial_messages(true)
        .extra_args(["--no-session-persistence"]);
    let config = ClaudeAcpConfig::new()
        .spawn(spawn)
        .default_mode("read-only");

    let (_peer, server_conn) = serve(server_transport, config);
    // The server must be polled before `connect` can complete `initialize`.
    let server_task = tokio::spawn(server_conn);
    let (client, client_conn) = Client::builder(Implementation::new("live-test", "0.0.0"))
        .permissions(|request: PermissionRequest| async move {
            let allow_id = request
                .options
                .iter()
                .find(|opt| matches!(&opt.kind, Open::Known(PermissionOptionKind::AllowOnce)))
                .map(|opt| opt.option_id.clone());
            allow_id.map_or_else(
                || PermissionOutcome::Cancelled(CancelledOutcome { meta: None }),
                |option_id| {
                    PermissionOutcome::Selected(SelectedOutcome {
                        option_id,
                        meta: None,
                    })
                },
            )
        })
        .connect(client_transport)
        .await
        .expect("ACP connect");

    let client_task = tokio::spawn(client_conn);

    let cwd = std::env::temp_dir();
    let session = tokio::time::timeout(Duration::from_secs(30), client.new_session(&cwd).open())
        .await
        .expect("session/new timed out")
        .expect("session/new failed");

    {
        let mut events = session.subscribe();
        let sink = Arc::clone(&chunks);
        tokio::spawn(async move {
            use aither_acp::SessionEvent;
            use futures_lite::StreamExt;
            while let Some(ev) = events.next().await {
                if let SessionEvent::MessageDelta { text: Some(t), .. } = ev {
                    sink.lock().unwrap().push(t);
                }
            }
        });
    }

    let outcome = tokio::time::timeout(
        Duration::from_secs(60),
        session.prompt_run(vec![ContentBlock::Text(TextContent {
            text: "Reply with the exact word PONG and nothing else.".to_string(),
            annotations: None,
            meta: None,
            extra: BTreeMap::default(),
        })]),
    )
    .await
    .expect("session/prompt timed out")
    .expect("session/prompt failed");

    let TurnOutcome::Finished { stop_reason, .. } = outcome else {
        panic!("expected Finished, got {outcome:?}");
    };
    assert!(matches!(
        stop_reason,
        Some(Open::Known(StopReason::EndTurn))
    ));

    let combined = chunks.lock().unwrap().join("");
    assert!(
        combined.to_uppercase().contains("PONG"),
        "expected PONG in streamed chunks, got: {combined:?}"
    );

    session.close().await.expect("session/close failed");
    client.close().await;

    tokio::time::timeout(Duration::from_secs(10), client_task)
        .await
        .expect("client shutdown timed out")
        .expect("client task panicked");
    tokio::time::timeout(Duration::from_secs(10), server_task)
        .await
        .expect("server shutdown timed out")
        .expect("server task panicked");
}
