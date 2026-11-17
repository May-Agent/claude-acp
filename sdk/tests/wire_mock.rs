//! End-to-end mock-CLI wire tests over an in-memory duplex byte pipe (plus
//! child-process exit propagation when `feature = "process"` is enabled).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use async_channel::{Receiver, Sender};
use claude_acp_sdk::{
    Client, ControlCancelRequest, ControlHandler, EffortLevel, HookCallbackRequest,
    McpMessageRequest, NdjsonReader, NdjsonWriter, NoopControlHandler, Permission,
    PermissionDecision, PermissionDestination, PermissionMode, PermissionModeName,
    RewindFilesRequest, SdkError, SessionEvent, StreamDelta, ToolPermissionRequest,
};
use futures_lite::io::{AsyncRead, AsyncWrite};
use futures_lite::stream::Stream;
use serde_json::{Value, json};

struct DuplexEnd {
    rx: Pin<Box<Receiver<Vec<u8>>>>,
    buf: Vec<u8>,
    pos: usize,
}

fn duplex_pair() -> (
    (DuplexRead, DuplexWrite),
    (NdjsonReader<DuplexRead>, NdjsonWriter<DuplexWrite>),
) {
    let (client_to_cli_tx, client_to_cli_rx) = async_channel::unbounded();
    let (cli_to_client_tx, cli_to_client_rx) = async_channel::unbounded();

    let client_read = DuplexRead(DuplexEnd {
        rx: Box::pin(cli_to_client_rx),
        buf: Vec::new(),
        pos: 0,
    });
    let client_write = DuplexWrite(client_to_cli_tx);

    let cli_read = DuplexRead(DuplexEnd {
        rx: Box::pin(client_to_cli_rx),
        buf: Vec::new(),
        pos: 0,
    });
    let cli_write = DuplexWrite(cli_to_client_tx);

    (
        (client_read, client_write),
        (NdjsonReader::new(cli_read), NdjsonWriter::new(cli_write)),
    )
}

struct DuplexRead(DuplexEnd);
struct DuplexWrite(Sender<Vec<u8>>);

impl AsyncRead for DuplexRead {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = &mut self.0;
        if this.pos < this.buf.len() {
            let rem = &this.buf[this.pos..];
            let count = rem.len().min(out.len());
            out[..count].copy_from_slice(&rem[..count]);
            this.pos += count;
            if this.pos == this.buf.len() {
                this.buf.clear();
                this.pos = 0;
            }
            return Poll::Ready(Ok(count));
        }

        match this.rx.as_mut().poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => Poll::Ready(Ok(0)),
            Poll::Ready(Some(chunk)) => {
                let count = chunk.len().min(out.len());
                out[..count].copy_from_slice(&chunk[..count]);
                if count < chunk.len() {
                    this.buf = chunk;
                    this.pos = count;
                }
                Poll::Ready(Ok(count))
            }
        }
    }
}

impl AsyncWrite for DuplexWrite {
    fn poll_write(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        self.0
            .try_send(buf.to_vec())
            .map(|()| buf.len())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "duplex closed"))
            .into()
    }

    fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Poll::Ready(Ok(()))
    }

    fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        self.0.close();
        Poll::Ready(Ok(()))
    }
}

async fn read_json_frame(cli_in: &mut NdjsonReader<DuplexRead>) -> Value {
    let line = cli_in
        .next_line()
        .await
        .expect("read frame")
        .expect("unexpected EOF");
    serde_json::from_str(&line).expect("valid JSON frame")
}

async fn send_control_ack(cli_out: &mut NdjsonWriter<DuplexWrite>, req_id: &Value, body: Value) {
    cli_out
        .send_json(&json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": req_id,
                "response": body
            }
        }))
        .await
        .unwrap();
}

#[tokio::test]
async fn initialize_handshake_and_control_error_roundtrip() {
    let ((c_read, c_write), (mut cli_in, mut cli_out)) = duplex_pair();
    let (client, driver) = Client::attach(c_read, c_write, NoopControlHandler);
    let driver_task = tokio::spawn(driver);

    let mock_cli = tokio::spawn(async move {
        let init_req = read_json_frame(&mut cli_in).await;
        assert_eq!(init_req["type"], "control_request");
        assert_eq!(init_req["request"]["subtype"], "initialize");
        send_control_ack(
            &mut cli_out,
            &init_req["request_id"],
            json!({
                "commands": [{"name": "compact", "description": "Compact context", "argumentHint": ""}],
                "agents": [],
                "output_style": "default",
                "available_output_styles": ["default"],
                "models": [{"value": "sonnet", "displayName": "Sonnet"}]
            }),
        )
        .await;

        let bad_req = read_json_frame(&mut cli_in).await;
        cli_out
            .send_json(&json!({
                "type": "control_response",
                "response": {
                    "subtype": "error",
                    "request_id": bad_req["request_id"],
                    "error": "unknown model requested"
                }
            }))
            .await
            .unwrap();
    });

    let init_res = client.initialize().await.unwrap();
    assert_eq!(init_res.commands.len(), 1);
    assert_eq!(init_res.commands[0].name, "compact");
    assert_eq!(init_res.output_style.as_deref(), Some("default"));

    let err = client.set_model(Some("nonexistent")).await.unwrap_err();
    assert!(matches!(
        err,
        SdkError::Control { message, .. } if message == "unknown model requested"
    ));

    mock_cli.await.unwrap();
    client.close();
    driver_task.await.unwrap().unwrap();
}

async fn run_turn_1_mock(
    cli_in: &mut NdjsonReader<DuplexRead>,
    cli_out: &mut NdjsonWriter<DuplexWrite>,
    tricky_prompt: &'static str,
    tricky_reply: &'static str,
) {
    let raw_user_1 = cli_in.next_line().await.unwrap().unwrap();
    assert!(raw_user_1.contains('\u{2028}') && raw_user_1.contains('\u{2029}'));
    let user_1: Value = serde_json::from_str(&raw_user_1).unwrap();
    assert_eq!(user_1["message"]["content"][0]["text"], tricky_prompt);

    cli_out
        .send_json(&json!({
            "type": "system",
            "subtype": "init",
            "session_id": "sess-multi",
            "cwd": "/tmp/project",
            "model": "claude-sonnet-4-6"
        }))
        .await
        .unwrap();

    cli_out
        .send_json(&json!({
            "type": "stream_event",
            "uuid": "ev-1",
            "session_id": "sess-multi",
            "parent_tool_use_id": null,
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "analyzing..." }
            }
        }))
        .await
        .unwrap();

    let stream_text_line = serde_json::to_string(&json!({
        "type": "stream_event",
        "uuid": "ev-2",
        "session_id": "sess-multi",
        "parent_tool_use_id": null,
        "event": {
            "type": "content_block_delta",
            "index": 1,
            "delta": { "type": "text_delta", "text": tricky_reply }
        }
    }))
    .unwrap();
    assert!(stream_text_line.contains('\u{2028}') && stream_text_line.contains('\u{2029}'));
    cli_out.write_raw_line(&stream_text_line).await.unwrap();

    cli_out
        .send_json(&json!({
            "type": "assistant",
            "session_id": "sess-multi",
            "message": {
                "id": "msg_1",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [
                    { "type": "thinking", "thinking": "analyzing...", "signature": "sig-1" },
                    { "type": "text", "text": tricky_reply }
                ]
            }
        }))
        .await
        .unwrap();

    cli_out
        .send_json(&json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "duration_ms": 42,
            "duration_api_ms": 35,
            "num_turns": 1,
            "session_id": "sess-multi",
            "total_cost_usd": 0.002
        }))
        .await
        .unwrap();
}

async fn run_multi_turn_mock_cli(
    mut cli_in: NdjsonReader<DuplexRead>,
    mut cli_out: NdjsonWriter<DuplexWrite>,
    tricky_prompt: &'static str,
    tricky_reply: &'static str,
) {
    run_turn_1_mock(&mut cli_in, &mut cli_out, tricky_prompt, tricky_reply).await;

    let user_2 = read_json_frame(&mut cli_in).await;
    assert_eq!(user_2["message"]["content"][0]["text"], "Turn 2");

    cli_out
        .send_json(&json!({
            "type": "stream_event",
            "uuid": "ev-3",
            "session_id": "sess-multi",
            "parent_tool_use_id": null,
            "event": {
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {} }
            }
        }))
        .await
        .unwrap();

    cli_out
        .send_json(&json!({
            "type": "stream_event",
            "uuid": "ev-4",
            "session_id": "sess-multi",
            "parent_tool_use_id": null,
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "input_json_delta", "partial_json": "{\"command\":\"pwd\"}" }
            }
        }))
        .await
        .unwrap();

    cli_out
        .send_json(&json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "duration_ms": 20,
            "duration_api_ms": 15,
            "num_turns": 2,
            "session_id": "sess-multi",
            "total_cost_usd": 0.004
        }))
        .await
        .unwrap();
}

#[tokio::test]
async fn multi_turn_streaming_deltas_and_unescaped_u2028_u2029() {
    let ((c_read, c_write), (cli_in, cli_out)) = duplex_pair();
    let (client, driver) = Client::attach(c_read, c_write, NoopControlHandler);
    let driver_task = tokio::spawn(driver);

    let tricky_prompt = "line1\u{2028}line2\u{2029}line3";
    let tricky_reply = "replyA\u{2028}replyB\u{2029}replyC";
    let mock_cli = tokio::spawn(run_multi_turn_mock_cli(
        cli_in,
        cli_out,
        tricky_prompt,
        tricky_reply,
    ));

    client.send_user_message(tricky_prompt).await.unwrap();

    let ev_init = client.next_event().await.unwrap().unwrap();
    assert!(matches!(ev_init, SessionEvent::System(ref s) if s.is_init()));

    let ev_think = client.next_event().await.unwrap().unwrap();
    assert_eq!(
        ev_think.as_stream_delta(),
        Some(StreamDelta::Thinking {
            index: 0,
            thinking: "analyzing...".to_string(),
        })
    );

    let ev_text = client.next_event().await.unwrap().unwrap();
    assert_eq!(
        ev_text.as_stream_delta(),
        Some(StreamDelta::Text {
            index: 1,
            text: tricky_reply.to_string(),
        })
    );

    let ev_assistant = client.next_event().await.unwrap().unwrap();
    let SessionEvent::Assistant(assistant_msg) = ev_assistant else {
        panic!("expected Assistant event");
    };
    assert_eq!(assistant_msg.session_id, "sess-multi");

    let ev_res_1 = client.next_event().await.unwrap().unwrap();
    assert!(matches!(ev_res_1, SessionEvent::Result(ref r) if r.num_turns == 1));

    client.send_user_message("Turn 2").await.unwrap();

    let ev_tool_start = client.next_event().await.unwrap().unwrap();
    assert_eq!(
        ev_tool_start.as_stream_delta(),
        Some(StreamDelta::ToolUseStart {
            index: 0,
            id: "toolu_1".to_string(),
            name: "Bash".to_string(),
        })
    );

    let ev_tool_json = client.next_event().await.unwrap().unwrap();
    assert_eq!(
        ev_tool_json.as_stream_delta(),
        Some(StreamDelta::ToolInputJson {
            index: 0,
            partial_json: "{\"command\":\"pwd\"}".to_string(),
        })
    );

    let ev_res_2 = client.next_event().await.unwrap().unwrap();
    assert!(matches!(ev_res_2, SessionEvent::Result(ref r) if r.num_turns == 2));

    mock_cli.await.unwrap();
    driver_task.await.unwrap().unwrap();
    assert!(client.next_event().await.unwrap().is_none());
}

struct PolicyHandler;

impl ControlHandler for PolicyHandler {
    fn can_use_tool(
        &self,
        req: ToolPermissionRequest,
    ) -> impl Future<Output = claude_acp_sdk::Result<PermissionDecision>> + Send {
        let decision = if req.tool_name == "Write" {
            PermissionDecision::allow_with_input(json!({
                "file_path": "/sandbox/out.txt",
                "content": "sanitized"
            }))
            .with_permissions([
                Permission::allow_tool("Write", "/sandbox/**"),
                Permission::set_mode(
                    PermissionModeName::AcceptEdits,
                    PermissionDestination::Session,
                ),
            ])
        } else {
            PermissionDecision::deny_and_interrupt("destructive command blocked")
        };
        std::future::ready(Ok(decision))
    }

    fn hook_callback(
        &self,
        req: HookCallbackRequest,
    ) -> impl Future<Output = claude_acp_sdk::Result<Value>> + Send {
        std::future::ready(Ok(
            json!({ "continue": true, "callbackId": req.callback_id }),
        ))
    }

    fn mcp_message(
        &self,
        req: McpMessageRequest,
    ) -> impl Future<Output = claude_acp_sdk::Result<Value>> + Send {
        std::future::ready(Ok(json!({
            "jsonrpc": "2.0",
            "id": req.message["id"],
            "result": { "server": req.server_name, "tools": [] }
        })))
    }
}

async fn run_permissions_mock_cli(
    mut cli_in: NdjsonReader<DuplexRead>,
    mut cli_out: NdjsonWriter<DuplexWrite>,
) {
    let _prompt = read_json_frame(&mut cli_in).await;

    cli_out
        .send_json(&json!({
            "type": "control_request",
            "request_id": "perm-allow-1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Write",
                "input": { "file_path": "/etc/hosts", "content": "raw" },
                "permission_suggestions": [],
                "tool_use_id": "toolu_w1"
            }
        }))
        .await
        .unwrap();

    let allow_reply = read_json_frame(&mut cli_in).await;
    assert_eq!(allow_reply["response"]["request_id"], "perm-allow-1");
    let allow_body = &allow_reply["response"]["response"];
    assert_eq!(allow_body["behavior"], "allow");
    assert_eq!(allow_body["updatedInput"]["file_path"], "/sandbox/out.txt");
    assert_eq!(
        allow_body["updatedPermissions"].as_array().unwrap().len(),
        2
    );

    cli_out
        .send_json(&json!({
            "type": "control_request",
            "request_id": "hook-1",
            "request": {
                "subtype": "hook_callback",
                "callback_id": "cb_pre_tool",
                "input": { "tool": "Bash" }
            }
        }))
        .await
        .unwrap();
    let hook_reply = read_json_frame(&mut cli_in).await;
    assert_eq!(
        hook_reply["response"]["response"]["callbackId"],
        "cb_pre_tool"
    );

    cli_out
        .send_json(&json!({
            "type": "control_request",
            "request_id": "mcp-1",
            "request": {
                "subtype": "mcp_message",
                "server_name": "local-sdk",
                "message": { "jsonrpc": "2.0", "id": 7, "method": "tools/list" }
            }
        }))
        .await
        .unwrap();
    let mcp_reply = read_json_frame(&mut cli_in).await;
    assert_eq!(
        mcp_reply["response"]["response"]["mcp_response"]["result"]["server"],
        "local-sdk"
    );

    cli_out
        .send_json(&json!({
            "type": "control_request",
            "request_id": "perm-deny-1",
            "request": {
                "subtype": "can_use_tool",
                "tool_name": "Bash",
                "input": { "command": "rm -rf /" },
                "permission_suggestions": []
            }
        }))
        .await
        .unwrap();
    let deny_reply = read_json_frame(&mut cli_in).await;
    let deny_body = &deny_reply["response"]["response"];
    assert_eq!(deny_body["behavior"], "deny");
    assert_eq!(deny_body["message"], "destructive command blocked");
    assert_eq!(deny_body["interrupt"], true);

    cli_out
        .send_json(&json!({
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": true,
            "duration_ms": 10,
            "duration_api_ms": 5,
            "num_turns": 1,
            "session_id": "sess-perm",
            "total_cost_usd": 0.001
        }))
        .await
        .unwrap();
}

#[tokio::test]
async fn mid_turn_can_use_tool_allow_deny_hooks_and_mcp_message() {
    let ((c_read, c_write), (cli_in, cli_out)) = duplex_pair();
    let (client, driver) = Client::attach(c_read, c_write, PolicyHandler);
    let driver_task = tokio::spawn(driver);
    let mock_cli = tokio::spawn(run_permissions_mock_cli(cli_in, cli_out));

    client.send_user_message("run tools").await.unwrap();
    let res = client.next_event().await.unwrap().unwrap();
    assert!(matches!(res, SessionEvent::Result(ref r) if r.is_error));

    mock_cli.await.unwrap();
    driver_task.await.unwrap().unwrap();
}

struct BlockingPermissionHandler {
    entered_tx: Sender<()>,
    cancelled: Arc<AtomicBool>,
}

impl ControlHandler for BlockingPermissionHandler {
    async fn can_use_tool(
        &self,
        _req: ToolPermissionRequest,
    ) -> claude_acp_sdk::Result<PermissionDecision> {
        let _ = self.entered_tx.send(()).await;
        std::future::pending::<claude_acp_sdk::Result<PermissionDecision>>().await
    }

    fn cancel_request(&self, _req: ControlCancelRequest) -> impl Future<Output = ()> + Send {
        self.cancelled.store(true, Ordering::SeqCst);
        std::future::ready(())
    }
}

#[tokio::test]
async fn mid_turn_interrupt_does_not_block_behind_parked_can_use_tool() {
    let (entered_tx, entered_rx) = async_channel::bounded(1);
    let cancelled = Arc::new(AtomicBool::new(false));
    let handler = BlockingPermissionHandler {
        entered_tx,
        cancelled: Arc::clone(&cancelled),
    };

    let ((c_read, c_write), (mut cli_in, mut cli_out)) = duplex_pair();
    let (client, driver) = Client::attach(c_read, c_write, handler);
    let driver_task = tokio::spawn(driver);

    let mock_cli = tokio::spawn(async move {
        let _prompt = read_json_frame(&mut cli_in).await;

        cli_out
            .send_json(&json!({
                "type": "control_request",
                "request_id": "perm-parked",
                "request": {
                    "subtype": "can_use_tool",
                    "tool_name": "Bash",
                    "input": { "command": "sleep 60" },
                    "permission_suggestions": []
                }
            }))
            .await
            .unwrap();

        let interrupt_frame = read_json_frame(&mut cli_in).await;
        assert_eq!(interrupt_frame["request"]["subtype"], "interrupt");

        cli_out
            .send_json(&json!({
                "type": "control_cancel_request",
                "request_id": "perm-parked"
            }))
            .await
            .unwrap();

        send_control_ack(&mut cli_out, &interrupt_frame["request_id"], json!({})).await;

        cli_out
            .send_json(&json!({
                "type": "result",
                "subtype": "success",
                "is_error": false,
                "duration_ms": 15,
                "duration_api_ms": 10,
                "num_turns": 1,
                "session_id": "sess-int",
                "total_cost_usd": 0.0
            }))
            .await
            .unwrap();
    });

    client.send_user_message("long task").await.unwrap();
    entered_rx.recv().await.unwrap();
    client.interrupt().await.unwrap();

    let result_ev = client.next_event().await.unwrap().unwrap();
    assert!(matches!(result_ev, SessionEvent::Result(_)));

    mock_cli.await.unwrap();
    driver_task.await.unwrap().unwrap();
    assert!(cancelled.load(Ordering::SeqCst));
}

async fn run_session_controls_mock_cli(
    mut cli_in: NdjsonReader<DuplexRead>,
    mut cli_out: NdjsonWriter<DuplexWrite>,
) {
    let f1 = read_json_frame(&mut cli_in).await;
    assert_eq!(f1["request"]["subtype"], "set_permission_mode");
    assert_eq!(f1["request"]["mode"], "plan");
    send_control_ack(&mut cli_out, &f1["request_id"], json!({ "mode": "plan" })).await;

    let f2 = read_json_frame(&mut cli_in).await;
    assert_eq!(f2["request"]["subtype"], "set_model");
    assert_eq!(f2["request"]["model"], "opus");
    send_control_ack(&mut cli_out, &f2["request_id"], json!({ "model": "opus" })).await;

    let f3 = read_json_frame(&mut cli_in).await;
    assert_eq!(f3["request"]["subtype"], "set_max_thinking_tokens");
    assert_eq!(f3["request"]["max_thinking_tokens"], 4096);
    send_control_ack(&mut cli_out, &f3["request_id"], json!({})).await;

    let f4 = read_json_frame(&mut cli_in).await;
    assert_eq!(f4["request"]["subtype"], "apply_flag_settings");
    assert_eq!(f4["request"]["settings"]["effortLevel"], "xhigh");
    send_control_ack(&mut cli_out, &f4["request_id"], json!({})).await;

    let f5 = read_json_frame(&mut cli_in).await;
    assert_eq!(f5["request"]["subtype"], "rewind_files");
    assert_eq!(f5["request"]["user_message_id"], "msg-checkpoint-1");
    assert_eq!(f5["request"]["dry_run"], true);
    send_control_ack(
        &mut cli_out,
        &f5["request_id"],
        json!({
            "canRewind": true,
            "filesChanged": ["src/lib.rs"],
            "insertions": 12,
            "deletions": 3
        }),
    )
    .await;

    let f6 = read_json_frame(&mut cli_in).await;
    assert_eq!(f6["request"]["subtype"], "mcp_status");
    send_control_ack(
        &mut cli_out,
        &f6["request_id"],
        json!({
            "mcpServers": [
                { "name": "filesystem", "status": "connected" },
                null
            ]
        }),
    )
    .await;
}

#[tokio::test]
async fn session_control_methods_set_mode_model_thinking_effort_rewind_and_mcp_status() {
    let ((c_read, c_write), (cli_in, cli_out)) = duplex_pair();
    let (client, driver) = Client::attach(c_read, c_write, NoopControlHandler);
    let driver_task = tokio::spawn(driver);
    let mock_cli = tokio::spawn(run_session_controls_mock_cli(cli_in, cli_out));

    client
        .set_permission_mode(PermissionMode::Plan)
        .await
        .unwrap();
    client.set_model(Some("opus")).await.unwrap();
    client.set_max_thinking_tokens(Some(4096)).await.unwrap();
    client.set_effort(EffortLevel::Xhigh).await.unwrap();

    let rewind = client
        .rewind_files_with_options(RewindFilesRequest::new("msg-checkpoint-1").dry_run(true))
        .await
        .unwrap();
    assert!(rewind.can_rewind);
    assert_eq!(rewind.files_changed, Some(vec!["src/lib.rs".to_string()]));
    assert_eq!(rewind.insertions, Some(12));
    assert_eq!(rewind.deletions, Some(3));

    let mcp = client.mcp_status().await.unwrap();
    assert_eq!(mcp.mcp_servers.len(), 1);
    assert_eq!(mcp.mcp_servers[0].name, "filesystem");
    assert_eq!(mcp.mcp_servers[0].status, "connected");

    mock_cli.await.unwrap();
    driver_task.await.unwrap().unwrap();
}

#[tokio::test]
async fn unexpected_eof_fails_all_pending_control_requests_immediately() {
    let ((c_read, c_write), (mut cli_in, mut cli_out)) = duplex_pair();
    let (client, driver) = Client::attach(c_read, c_write, NoopControlHandler);
    let driver_task = tokio::spawn(driver);

    let c1 = client.clone();
    let c2 = client.clone();
    let req1 = tokio::spawn(async move { c1.initialize().await });
    let req2 = tokio::spawn(async move { c2.mcp_status().await });

    let _ = read_json_frame(&mut cli_in).await;
    let _ = read_json_frame(&mut cli_in).await;
    cli_out.close().await.unwrap();
    drop(cli_out);

    assert!(matches!(req1.await.unwrap(), Err(SdkError::ChannelClosed)));
    assert!(matches!(req2.await.unwrap(), Err(SdkError::ChannelClosed)));
    assert!(client.next_event().await.unwrap().is_none());

    driver_task.await.unwrap().unwrap();
}

#[cfg(feature = "process")]
#[tokio::test]
async fn child_process_nonzero_exit_propagates_status_and_stderr_to_pending_callers() {
    use claude_acp_sdk::SessionConfig;

    // `/bin/sh` rejects `--print` with a non-zero exit code and diagnostic stderr.
    let cfg = SessionConfig::new().claude_path("/bin/sh");

    let (client, driver) = Client::spawn(cfg, NoopControlHandler).unwrap();
    let driver_task = tokio::spawn(driver);

    let init_err = client.initialize().await.unwrap_err();
    match &init_err {
        SdkError::ProcessExited { status, stderr } => {
            assert!(status.is_some_and(|s| !s.success()));
            assert!(!stderr.trim().is_empty(), "expected non-empty stderr");
        }
        other => panic!("expected ProcessExited, got {other:?}"),
    }

    let event_err = client.next_event().await.unwrap_err();
    assert!(matches!(event_err, SdkError::ProcessExited { .. }));

    let driver_err = driver_task.await.unwrap().unwrap_err();
    assert!(matches!(driver_err, SdkError::ProcessExited { .. }));
}

#[cfg(all(feature = "process", unix))]
#[tokio::test]
async fn client_close_on_spawned_child_resolves_cleanly() {
    use std::os::unix::fs::PermissionsExt;

    use claude_acp_sdk::SessionConfig;
    use uuid::Uuid;

    let script_dir = std::env::temp_dir().join(format!("claude-acp-sdk-close-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&script_dir).unwrap();
    let script_path = script_dir.join("mock-claude.sh");
    std::fs::write(&script_path, "#!/bin/sh\nsleep 60\n").unwrap();
    std::fs::set_permissions(&script_path, std::fs::Permissions::from_mode(0o755)).unwrap();

    let cfg = SessionConfig::new().claude_path(&script_path);
    let (client, driver) = Client::spawn(cfg, NoopControlHandler).unwrap();
    let driver_task = tokio::spawn(driver);

    client.close();

    assert!(
        client.next_event().await.unwrap().is_none(),
        "explicit close must end event stream cleanly with Ok(None)"
    );
    let status = driver_task
        .await
        .unwrap()
        .expect("explicit close must resolve driver with Ok(ExitStatus)");
    let _ = status;

    let _ = std::fs::remove_dir_all(&script_dir);
}
