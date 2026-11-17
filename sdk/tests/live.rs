//! Live smoke test exercising a real `claude` CLI binary (`cargo test -- --ignored`).

#![cfg(feature = "process")]

use std::time::Duration;

use claude_acp_sdk::{
    ClaudeModel, Client, NoopControlHandler, PermissionMode, SessionConfig, SessionEvent,
};

#[tokio::test]
#[ignore = "requires an authenticated local `claude` CLI installation"]
async fn live_claude_cli_round_trip() {
    let config = SessionConfig::new()
        .model(ClaudeModel::Haiku)
        .permission_mode(PermissionMode::Plan)
        .include_partial_messages(true)
        .extra_args(["--no-session-persistence"]);

    let (client, driver) = Client::spawn(config, NoopControlHandler).expect("spawn claude");
    let driver_task = tokio::spawn(driver);

    let init = tokio::time::timeout(Duration::from_secs(30), client.initialize())
        .await
        .expect("initialize timed out")
        .expect("initialize failed");
    assert!(!init.commands.is_empty());

    client
        .send_user_message("Reply with the exact word PONG and nothing else.")
        .await
        .expect("send prompt");

    let mut saw_init = false;
    let mut saw_assistant = false;
    let mut saw_result = false;

    while let Some(event) = tokio::time::timeout(Duration::from_secs(60), client.next_event())
        .await
        .expect("event stream timed out")
        .expect("stream error")
    {
        match event {
            SessionEvent::System(sys) if sys.is_init() => saw_init = true,
            SessionEvent::Assistant(_) => saw_assistant = true,
            SessionEvent::Result(res) => {
                assert!(!res.is_error, "unexpected error result: {res:?}");
                saw_result = true;
                break;
            }
            _ => {}
        }
    }

    assert!(saw_init, "expected system/init event");
    assert!(saw_assistant, "expected assistant message");
    assert!(saw_result, "expected terminal result message");

    client.close();
    tokio::time::timeout(Duration::from_secs(10), driver_task)
        .await
        .expect("driver shutdown timed out")
        .expect("driver task panicked")
        .expect("clean driver shutdown");
}
