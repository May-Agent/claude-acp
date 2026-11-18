//! End-to-end integration tests pairing `aither_acp::AcpClient` with
//! `claude_acp::serve` over `DuplexTransport` against an in-memory mock Claude
//! CLI NDJSON stream.

use std::collections::VecDeque;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use aither_acp::ext::auth_status::{self, AuthStatus, AuthStatusKind};
use aither_acp::{
    AcpClient, AuthenticateParams, ClientAuthCapabilities, ClientCapabilities, ClientHandler,
    ConfigOptionValue, ContentBlock, EmbeddedResource, EnvVar, HttpHeader, ImageContent,
    LogoutParams, McpServer, McpServerHttp, McpServerSse, McpServerStdio, PROTOCOL_VERSION,
    PermissionOptionKind, PlanEntryStatus, PromptParams, RequestPermissionOutcome,
    RequestPermissionParams, RequestPermissionResult, ResourceContent, ResourceLink,
    SessionCancelParams, SessionCloseParams, SessionDeleteParams, SessionListParams,
    SessionLoadParams, SessionNewParams, SessionNotification, SessionResumeParams,
    SessionSetConfigOptionParams, SessionSetModeParams, SessionUpdate, StopReason, TextContent,
    ToolCallContent, ToolCallStatus, ToolKind,
};
use aither_mcp::protocol::{JsonRpcError, JsonRpcNotification};
use aither_mcp::transport::DuplexTransport;
use async_channel::{Receiver, Sender};
use claude_acp::sdk::{NdjsonReader, NdjsonWriter, SpawnConfig, transcript_path};
use claude_acp::{
    AUTH_METHOD_API_KEY, AUTH_METHOD_CLAUDE_LOGIN, ClaudeAcpConfig, ConnectedSession, serve,
};
use futures_lite::io::{AsyncRead, AsyncWrite};
use futures_lite::stream::Stream;
use serde_json::{Value, json};
use uuid::Uuid;

// =============================================================================
// In-memory byte pipe & Mock CLI harness
// =============================================================================

struct BytePipeRead {
    rx: Pin<Box<Receiver<Vec<u8>>>>,
    buf: Vec<u8>,
    pos: usize,
}

struct BytePipeWrite(Sender<Vec<u8>>);

fn mock_cli_pipe() -> (
    (BytePipeRead, BytePipeWrite),
    (NdjsonReader<BytePipeRead>, NdjsonWriter<BytePipeWrite>),
) {
    let (sdk_to_cli_tx, sdk_to_cli_rx) = async_channel::unbounded();
    let (cli_to_sdk_tx, cli_to_sdk_rx) = async_channel::unbounded();

    let sdk_read = BytePipeRead {
        rx: Box::pin(cli_to_sdk_rx),
        buf: Vec::new(),
        pos: 0,
    };
    let sdk_write = BytePipeWrite(sdk_to_cli_tx);

    let cli_read = BytePipeRead {
        rx: Box::pin(sdk_to_cli_rx),
        buf: Vec::new(),
        pos: 0,
    };
    let cli_write = BytePipeWrite(cli_to_sdk_tx);

    (
        (sdk_read, sdk_write),
        (NdjsonReader::new(cli_read), NdjsonWriter::new(cli_write)),
    )
}

impl AsyncRead for BytePipeRead {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        out: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = &mut *self;
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

impl AsyncWrite for BytePipeWrite {
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
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "mock pipe closed"))
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

struct MockCliEndpoint {
    spawn_config: SpawnConfig,
    reader: NdjsonReader<BytePipeRead>,
    writer: NdjsonWriter<BytePipeWrite>,
}

impl MockCliEndpoint {
    async fn recv(&mut self) -> Value {
        let line = self
            .reader
            .next_line()
            .await
            .expect("mock CLI read")
            .expect("unexpected EOF on mock CLI stdin");
        serde_json::from_str(&line).expect("valid JSON frame")
    }

    async fn send(&mut self, frame: &Value) {
        self.writer.send_json(frame).await.expect("mock CLI write");
    }

    async fn ack_initialize(&mut self, extra_commands: Vec<Value>) -> Value {
        let req = self.recv().await;
        assert_eq!(req["type"], "control_request");
        assert_eq!(req["request"]["subtype"], "initialize");
        self.ack_control(
            &req["request_id"],
            json!({
                "commands": extra_commands,
                "agents": [],
                "models": [
                    { "value": "sonnet", "displayName": "Sonnet" },
                    { "value": "opus", "displayName": "Opus" }
                ]
            }),
        )
        .await;
        req
    }

    async fn ack_control(&mut self, req_id: &Value, body: Value) {
        self.send(&json!({
            "type": "control_response",
            "response": {
                "subtype": "success",
                "request_id": req_id,
                "response": body
            }
        }))
        .await;
    }
}

struct TempHome {
    path: PathBuf,
}

impl TempHome {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("claude-acp-e2e-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&path).expect("create temp home");
        Self { path }
    }
}

impl Drop for TempHome {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[derive(Clone, Copy)]
enum PermissionReply {
    SelectKind(PermissionOptionKind),
    Cancel,
}

#[derive(Clone)]
struct RecordingClientHandler {
    inner: Arc<RecordingInner>,
}

struct RecordingInner {
    updates: Mutex<Vec<SessionNotification>>,
    auth_updates: Mutex<Vec<AuthStatus>>,
    permission_requests: Mutex<Vec<RequestPermissionParams>>,
    permission_plan: Mutex<VecDeque<PermissionReply>>,
    terminal_auth: bool,
}

impl RecordingClientHandler {
    fn new(terminal_auth: bool) -> Self {
        Self {
            inner: Arc::new(RecordingInner {
                updates: Mutex::new(Vec::new()),
                auth_updates: Mutex::new(Vec::new()),
                permission_requests: Mutex::new(Vec::new()),
                permission_plan: Mutex::new(VecDeque::new()),
                terminal_auth,
            }),
        }
    }

    fn push_permission_reply(&self, reply: PermissionReply) {
        self.inner.permission_plan.lock().unwrap().push_back(reply);
    }

    fn take_updates(&self) -> Vec<SessionNotification> {
        std::mem::take(&mut *self.inner.updates.lock().unwrap())
    }

    fn take_auth_updates(&self) -> Vec<AuthStatus> {
        std::mem::take(&mut *self.inner.auth_updates.lock().unwrap())
    }

    fn permission_request_count(&self) -> usize {
        self.inner.permission_requests.lock().unwrap().len()
    }
}

impl ClientHandler for RecordingClientHandler {
    fn capabilities(&self) -> ClientCapabilities {
        ClientCapabilities {
            auth: Some(ClientAuthCapabilities {
                terminal: self.inner.terminal_auth,
                meta: None,
            }),
            ..ClientCapabilities::default()
        }
    }

    fn session_update(&self, notification: SessionNotification) -> impl Future<Output = ()> + Send {
        self.inner.updates.lock().unwrap().push(notification);
        std::future::ready(())
    }

    fn notification(&self, notification: JsonRpcNotification) -> impl Future<Output = ()> + Send {
        if let Some(status) = auth_status::notification(&notification) {
            self.inner.auth_updates.lock().unwrap().push(status);
        }
        std::future::ready(())
    }

    fn request_permission(
        &self,
        params: RequestPermissionParams,
    ) -> impl Future<Output = Result<RequestPermissionResult, JsonRpcError>> + Send {
        self.inner
            .permission_requests
            .lock()
            .unwrap()
            .push(params.clone());
        let next = self
            .inner
            .permission_plan
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(PermissionReply::Cancel);

        let outcome = match next {
            PermissionReply::Cancel => RequestPermissionOutcome::Cancelled,
            PermissionReply::SelectKind(target_kind) => params
                .options
                .iter()
                .find(|opt| opt.kind == target_kind)
                .map_or(RequestPermissionOutcome::Cancelled, |opt| {
                    RequestPermissionOutcome::Selected {
                        option_id: opt.option_id.clone(),
                    }
                }),
        };

        std::future::ready(Ok(RequestPermissionResult {
            outcome,
            meta: None,
        }))
    }
}

struct TestHarness {
    home: TempHome,
    handler: RecordingClientHandler,
    client: AcpClient<RecordingClientHandler>,
    cli_rx: Receiver<MockCliEndpoint>,
    server_task: tokio::task::JoinHandle<()>,
    client_task: tokio::task::JoinHandle<()>,
}

impl TestHarness {
    fn start(terminal_auth: bool, page_size: usize) -> Self {
        let home = TempHome::new();
        let handler = RecordingClientHandler::new(terminal_auth);
        let (cli_tx, cli_rx) = async_channel::unbounded();

        let config = ClaudeAcpConfig::new()
            .transcript_home(&home.path)
            .session_list_page_size(page_size)
            .connector(move |spawn_config, control_handler| {
                let cli_tx = cli_tx.clone();
                async move {
                    let ((sdk_r, sdk_w), (reader, writer)) = mock_cli_pipe();
                    let _ = cli_tx
                        .send(MockCliEndpoint {
                            spawn_config,
                            reader,
                            writer,
                        })
                        .await;
                    Ok(ConnectedSession::attach(sdk_r, sdk_w, control_handler))
                }
            });

        let (client_transport, server_transport) = DuplexTransport::pair();
        let (_peer, server_conn) = serve(server_transport, config);
        let (client, client_conn) = AcpClient::connect(client_transport, handler.clone());

        Self {
            home,
            handler,
            client,
            cli_rx,
            server_task: tokio::spawn(server_conn),
            client_task: tokio::spawn(client_conn),
        }
    }

    async fn shutdown(self) {
        self.client.close();
        tokio::time::timeout(Duration::from_secs(5), self.client_task)
            .await
            .expect("client_task exit")
            .expect("client_task join");
        tokio::time::timeout(Duration::from_secs(5), self.server_task)
            .await
            .expect("server_task exit")
            .expect("server_task join");
    }
}

fn text_block(text: &str) -> ContentBlock {
    ContentBlock::Text(TextContent {
        text: text.to_string(),
        annotations: None,
        meta: None,
    })
}

fn write_transcript_jsonl(path: &Path, lines: &[Value]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut raw = String::new();
    for item in lines {
        raw.push_str(&serde_json::to_string(item).unwrap());
        raw.push('\n');
    }
    std::fs::write(path, raw).unwrap();
}

// =============================================================================
// 1. Handshake, Capabilities, Auth, and `_auth/status_update`
// =============================================================================

#[tokio::test]
async fn initialize_capabilities_and_auth_status_round_trip() {
    let harness = TestHarness::start(true, 50);

    let init = harness.client.initialize().await.expect("initialize");
    assert_eq!(init.protocol_version, PROTOCOL_VERSION);
    assert!(init.agent_capabilities.load_session);
    assert!(init.agent_capabilities.prompt_capabilities.image);
    assert!(!init.agent_capabilities.prompt_capabilities.audio);
    assert!(init.agent_capabilities.prompt_capabilities.embedded_context);
    assert!(init.agent_capabilities.mcp_capabilities.http);
    assert!(init.agent_capabilities.mcp_capabilities.sse);
    assert!(init.agent_capabilities.session_capabilities.list.is_some());
    assert!(
        init.agent_capabilities
            .session_capabilities
            .delete
            .is_some()
    );
    assert!(init.agent_capabilities.session_capabilities.close.is_some());
    assert!(
        init.agent_capabilities
            .session_capabilities
            .resume
            .is_some()
    );
    assert!(
        init.agent_capabilities
            .session_capabilities
            .additional_directories
            .is_some()
    );
    assert!(auth_status::advertised(&init.agent_capabilities));
    assert!(
        init.agent_capabilities
            .auth
            .as_ref()
            .and_then(|a| a.logout.as_ref())
            .is_some()
    );

    let method_ids: Vec<&str> = init
        .auth_methods
        .iter()
        .map(aither_acp::AuthMethod::id)
        .collect();
    assert!(method_ids.contains(&AUTH_METHOD_API_KEY));
    assert!(method_ids.contains(&AUTH_METHOD_CLAUDE_LOGIN));

    // Authenticate with an API key and verify `_auth/status_update`.
    harness
        .client
        .authenticate(
            AuthenticateParams::new(AUTH_METHOD_API_KEY)
                .meta(json!({ "apiKey": "sk-ant-test-key-123" })),
        )
        .await
        .expect("authenticate api-key");

    // Authenticate with account metadata.
    harness
        .client
        .authenticate(
            AuthenticateParams::new(AUTH_METHOD_CLAUDE_LOGIN).meta(json!({
                "email": "dev@example.com",
                "organization": "May-Agent",
                "plan": "max"
            })),
        )
        .await
        .expect("authenticate claude-login");

    // Logout clears credentials and emits `AuthStatusKind::None`.
    harness
        .client
        .logout(LogoutParams::new())
        .await
        .expect("logout");

    let auth_events = harness.handler.take_auth_updates();
    assert_eq!(auth_events.len(), 3);
    assert_eq!(auth_events[0].kind, AuthStatusKind::ApiKey);
    assert_eq!(auth_events[1].kind, AuthStatusKind::Account);
    assert_eq!(
        auth_events[1]
            .account
            .as_ref()
            .and_then(|a| a.email.as_deref()),
        Some("dev@example.com")
    );
    assert_eq!(auth_events[2].kind, AuthStatusKind::None);

    harness.shutdown().await;
}

// =============================================================================
// 2. `session/new` forwarding, prompt conversions, streaming & de-duplication
// =============================================================================

async fn mock_streaming_turn(mut cli: MockCliEndpoint) {
    let _ = cli.ack_initialize(vec![]).await;

    let user_msg = cli.recv().await;
    let blocks = user_msg["message"]["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 4);
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[1]["type"], "image");
    assert_eq!(blocks[1]["source"]["media_type"], "image/png");
    assert!(blocks[2]["text"].as_str().unwrap().contains("fn main() {}"));
    assert!(blocks[3]["text"].as_str().unwrap().contains("spec.md"));

    cli.send(&json!({
        "type": "auth_status",
        "isAuthenticating": false,
        "output": ["Token refreshed"],
        "uuid": "auth-1",
        "session_id": "sess-stream"
    }))
    .await;

    emit_mock_stream_deltas_and_assistant(&mut cli).await;

    cli.send(&json!({
        "type": "user",
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "message": {
            "role": "user",
            "content": [
                { "type": "tool_result", "tool_use_id": "tu-todo", "content": "Todos updated" },
                { "type": "tool_result", "tool_use_id": "tu-edit", "content": "Updated src/main.rs", "is_error": false }
            ]
        }
    }))
    .await;

    cli.send(&json!({
        "type": "result",
        "subtype": "success",
        "is_error": false,
        "duration_ms": 25,
        "duration_api_ms": 20,
        "num_turns": 1,
        "session_id": "sess-stream",
        "total_cost_usd": 0.001
    }))
    .await;
}

async fn emit_mock_stream_deltas_and_assistant(cli: &mut MockCliEndpoint) {
    for (uuid, ev) in [
        (
            "se-0",
            json!({ "type": "message_start", "message": { "id": "msg_stream_1" } }),
        ),
        (
            "se-1",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "thinking_delta", "thinking": "Step 1: inspect." }
            }),
        ),
        (
            "se-2",
            json!({
                "type": "content_block_delta",
                "index": 1,
                "delta": { "type": "text_delta", "text": "Applying patch." }
            }),
        ),
    ] {
        cli.send(&json!({
            "type": "stream_event",
            "uuid": uuid,
            "session_id": "sess-stream",
            "parent_tool_use_id": null,
            "event": ev
        }))
        .await;
    }

    cli.send(&json!({
        "type": "assistant",
        "session_id": "sess-stream",
        "message": {
            "id": "msg_stream_1",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [
                { "type": "thinking", "thinking": "Step 1: inspect.", "signature": "sig" },
                { "type": "text", "text": "Applying patch." },
                {
                    "type": "tool_use",
                    "id": "tu-todo",
                    "name": "TodoWrite",
                    "input": {
                        "todos": [
                            { "content": "Update handler", "status": "in_progress", "activeForm": "Updating handler" },
                            { "content": "Run tests", "status": "pending", "activeForm": "Running tests" }
                        ]
                    }
                },
                {
                    "type": "tool_use",
                    "id": "tu-edit",
                    "name": "Edit",
                    "input": {
                        "file_path": "/workspace/src/main.rs",
                        "old_string": "old()",
                        "new_string": "new()"
                    }
                }
            ]
        }
    }))
    .await;
}

fn verify_streamed_turn_updates(updates: Vec<SessionNotification>) {
    let mut thoughts = Vec::new();
    let mut messages = Vec::new();
    let mut saw_plan = false;
    let mut saw_edit_diff = false;
    let mut saw_edit_completed = false;

    for notif in updates {
        match notif.update {
            SessionUpdate::AgentThoughtChunk(c) => {
                if let ContentBlock::Text(t) = c.content {
                    thoughts.push(t.text);
                }
            }
            SessionUpdate::AgentMessageChunk(c) => {
                if let ContentBlock::Text(t) = c.content {
                    messages.push(t.text);
                }
            }
            SessionUpdate::Plan(plan) => {
                assert_eq!(plan.entries.len(), 2);
                assert_eq!(plan.entries[0].status, PlanEntryStatus::InProgress);
                assert_eq!(plan.entries[1].status, PlanEntryStatus::Pending);
                saw_plan = true;
            }
            SessionUpdate::ToolCall(tc) => {
                assert_eq!(tc.tool_call_id, "tu-edit");
                assert_eq!(tc.kind, Some(ToolKind::Edit));
                assert_eq!(
                    tc.locations[0].path,
                    PathBuf::from("/workspace/src/main.rs")
                );
                assert!(matches!(
                    &tc.content[0],
                    ToolCallContent::Diff(d)
                        if d.old_text.as_deref() == Some("old()") && d.new_text == "new()"
                ));
                saw_edit_diff = true;
            }
            SessionUpdate::ToolCallUpdate(upd) => {
                assert_eq!(upd.tool_call_id, "tu-edit");
                assert_eq!(upd.status, Some(ToolCallStatus::Completed));
                saw_edit_completed = true;
            }
            _ => {}
        }
    }

    assert_eq!(thoughts, vec!["Step 1: inspect."]);
    assert_eq!(messages, vec!["Applying patch."]);
    assert!(saw_plan);
    assert!(saw_edit_diff);
    assert!(saw_edit_completed);
}

#[tokio::test]
async fn new_session_forwards_mcp_dirs_and_streams_deduplicated_updates() {
    let harness = TestHarness::start(false, 50);
    let _ = harness.client.initialize().await.expect("initialize");

    let cli_rx = harness.cli_rx.clone();
    let mock_task = tokio::spawn(async move {
        let cli = cli_rx.recv().await.unwrap();
        assert_eq!(
            cli.spawn_config.cwd.as_deref(),
            Some(Path::new("/workspace"))
        );
        assert_eq!(
            cli.spawn_config.additional_directories,
            vec![PathBuf::from("/extra-root")]
        );
        assert_eq!(cli.spawn_config.mcp_servers.len(), 3);
        mock_streaming_turn(cli).await;
    });

    let new_params = SessionNewParams::new("/workspace")
        .additional_directories(vec![PathBuf::from("/extra-root")])
        .mcp_servers(vec![
            McpServer::from(
                McpServerStdio::new("local-stdio", "mcp-bin")
                    .args(vec!["--serve".to_string()])
                    .env(vec![EnvVar {
                        name: "FOO".to_string(),
                        value: "bar".to_string(),
                    }]),
            ),
            McpServer::from(
                McpServerHttp::new("remote-http", "https://example.invalid/mcp")
                    .headers(vec![HttpHeader::new("Authorization", "Bearer tok")]),
            ),
            McpServer::from(McpServerSse::new(
                "remote-sse",
                "https://example.invalid/sse",
            )),
        ]);

    let created = harness.client.new_session(new_params).await.expect("new");
    assert_eq!(created.modes.expect("modes").current_mode_id, "agent");
    assert_eq!(created.config_options.expect("config_options").len(), 3);

    let init_updates = harness.handler.take_updates();
    assert!(init_updates.iter().any(|n| matches!(
        &n.update,
        SessionUpdate::AvailableCommandsUpdate(cmds)
            if ["compact", "clear", "status", "logout", "review"]
                .iter()
                .all(|req_cmd| cmds.available_commands.iter().any(|c| c.name == *req_cmd))
    )));

    let prompt_blocks = vec![
        text_block("Refactor main"),
        ContentBlock::Image(ImageContent {
            data: Some("iVBORw0KGgo=".to_string()),
            uri: None,
            mime_type: "image/png".to_string(),
            annotations: None,
            meta: None,
        }),
        ContentBlock::Resource(ResourceContent {
            resource: EmbeddedResource {
                uri: "file:///workspace/src/main.rs".to_string(),
                mime_type: Some("text/x-rust".to_string()),
                text: Some("fn main() {}".to_string()),
                blob: None,
                meta: None,
            },
            annotations: None,
            meta: None,
        }),
        ContentBlock::ResourceLink(ResourceLink::new("file:///workspace/spec.md", "spec.md")),
    ];

    let prompt_res = harness
        .client
        .prompt(PromptParams::new(&created.session_id, prompt_blocks))
        .await
        .expect("prompt");
    assert_eq!(prompt_res.stop_reason, StopReason::EndTurn);
    mock_task.await.unwrap();

    let auth_updates = harness.handler.take_auth_updates();
    assert_eq!(auth_updates.len(), 1);
    assert_eq!(auth_updates[0].detail.as_deref(), Some("Token refreshed"));

    verify_streamed_turn_updates(harness.handler.take_updates());
    harness.shutdown().await;
}

// =============================================================================
// 3. `can_use_tool` <-> `request_permission` (AllowOnce, AllowAlways, Reject, Cancel)
// =============================================================================

async fn send_can_use_tool(
    cli: &mut MockCliEndpoint,
    req_id: &str,
    tool_name: &str,
    input: Value,
    suggestions: Value,
) -> Value {
    cli.send(&json!({
        "type": "control_request",
        "request_id": req_id,
        "request": {
            "subtype": "can_use_tool",
            "tool_name": tool_name,
            "input": input,
            "permission_suggestions": suggestions,
            "tool_use_id": req_id
        }
    }))
    .await;
    cli.recv().await
}

async fn mock_permissions_cli(mut cli: MockCliEndpoint) {
    let _ = cli.ack_initialize(vec![]).await;
    let _ = cli.recv().await;

    let r1 = send_can_use_tool(
        &mut cli,
        "p-1",
        "Write",
        json!({ "file_path": "/workspace/a.txt", "content": "hello" }),
        json!([]),
    )
    .await;
    assert_eq!(r1["response"]["response"]["behavior"], "allow");

    let r2 = send_can_use_tool(
        &mut cli,
        "p-2",
        "Bash",
        json!({ "command": "cargo check" }),
        json!([{
            "type": "addRules",
            "destination": "session",
            "behavior": "allow",
            "rules": [{ "toolName": "Bash", "ruleContent": "cargo *" }]
        }]),
    )
    .await;
    assert_eq!(r2["response"]["response"]["behavior"], "allow");
    assert_eq!(
        r2["response"]["response"]["updatedPermissions"]
            .as_array()
            .unwrap()
            .len(),
        1
    );

    let r3 = send_can_use_tool(
        &mut cli,
        "p-3",
        "Bash",
        json!({ "command": "cargo test" }),
        json!([]),
    )
    .await;
    assert_eq!(r3["response"]["response"]["behavior"], "allow");

    let r4 = send_can_use_tool(
        &mut cli,
        "p-4",
        "WebFetch",
        json!({ "url": "https://example.invalid", "prompt": "read" }),
        json!([]),
    )
    .await;
    assert_eq!(r4["response"]["response"]["behavior"], "deny");
    assert!(
        !r4["response"]["response"]["interrupt"]
            .as_bool()
            .unwrap_or(false)
    );

    for req_id in ["p-5", "p-6"] {
        let rep = send_can_use_tool(
            &mut cli,
            req_id,
            "WebSearch",
            json!({ "query": "secret" }),
            json!([]),
        )
        .await;
        assert_eq!(rep["response"]["response"]["behavior"], "deny");
        assert!(
            !rep["response"]["response"]["interrupt"]
                .as_bool()
                .unwrap_or(false)
        );
    }

    cli.send(&json!({
        "type": "result",
        "subtype": "success",
        "is_error": false,
        "duration_ms": 10,
        "duration_api_ms": 5,
        "num_turns": 1,
        "session_id": "sess-perm",
        "total_cost_usd": 0.0
    }))
    .await;

    run_cancelled_permission_turn(&mut cli).await;
}

async fn run_cancelled_permission_turn(cli: &mut MockCliEndpoint) {
    let _ = cli.recv().await;
    let r_cancel = send_can_use_tool(
        cli,
        "p-cancel",
        "Write",
        json!({ "file_path": "/workspace/b.txt", "content": "x" }),
        json!([]),
    )
    .await;
    assert_eq!(r_cancel["response"]["response"]["behavior"], "deny");
    assert_eq!(r_cancel["response"]["response"]["interrupt"], true);

    cli.send(&json!({
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": true,
        "duration_ms": 5,
        "duration_api_ms": 2,
        "num_turns": 2,
        "session_id": "sess-perm",
        "total_cost_usd": 0.0
    }))
    .await;
}

#[tokio::test]
async fn permission_bridge_remembers_session_grants_rejections_and_cancellation() {
    let harness = TestHarness::start(false, 50);
    let _ = harness.client.initialize().await.expect("initialize");

    harness
        .handler
        .push_permission_reply(PermissionReply::SelectKind(PermissionOptionKind::AllowOnce));
    harness
        .handler
        .push_permission_reply(PermissionReply::SelectKind(
            PermissionOptionKind::AllowAlways,
        ));
    harness
        .handler
        .push_permission_reply(PermissionReply::SelectKind(
            PermissionOptionKind::RejectOnce,
        ));
    harness
        .handler
        .push_permission_reply(PermissionReply::SelectKind(
            PermissionOptionKind::RejectAlways,
        ));
    harness
        .handler
        .push_permission_reply(PermissionReply::Cancel);

    let cli_rx = harness.cli_rx.clone();
    let mock_task = tokio::spawn(async move {
        let cli = cli_rx.recv().await.unwrap();
        mock_permissions_cli(cli).await;
    });

    let session = harness
        .client
        .new_session(SessionNewParams::new("/workspace"))
        .await
        .expect("new_session");

    let turn1 = harness
        .client
        .prompt(PromptParams::new(
            &session.session_id,
            vec![text_block("Run tools")],
        ))
        .await
        .expect("turn 1");
    assert_eq!(turn1.stop_reason, StopReason::EndTurn);

    // 6 CLI `can_use_tool` requests occurred in turn 1, but only 4 prompted the client!
    assert_eq!(harness.handler.permission_request_count(), 4);

    let turn2 = harness
        .client
        .prompt(PromptParams::new(
            &session.session_id,
            vec![text_block("Trigger cancelled permission")],
        ))
        .await
        .expect("turn 2");
    assert_eq!(turn2.stop_reason, StopReason::Cancelled);

    mock_task.await.unwrap();
    harness.shutdown().await;
}

// =============================================================================
// 4. Mid-turn `session/cancel`, Modes, Config Options, and Slash Commands
// =============================================================================

async fn mock_cancel_modes_and_slash_cli(mut cli: MockCliEndpoint, started_tx: Sender<()>) {
    let _ = cli
        .ack_initialize(vec![json!({
            "name": "review",
            "description": "Code review",
            "argumentHint": "[path]"
        })])
        .await;

    let _ = cli.recv().await;
    let _ = started_tx.send(()).await;
    let interrupt_req = cli.recv().await;
    assert_eq!(interrupt_req["request"]["subtype"], "interrupt");
    cli.ack_control(&interrupt_req["request_id"], json!({}))
        .await;
    send_ok_result(&mut cli, 1, None).await;

    for expected_mode in ["plan", "acceptEdits", "bypassPermissions"] {
        let mode_req = cli.recv().await;
        assert_eq!(mode_req["request"]["subtype"], "set_permission_mode");
        assert_eq!(mode_req["request"]["mode"], expected_mode);
        cli.ack_control(&mode_req["request_id"], json!({})).await;
    }

    let model_req = cli.recv().await;
    assert_eq!(model_req["request"]["subtype"], "set_model");
    assert_eq!(model_req["request"]["model"], "opus");
    cli.ack_control(&model_req["request_id"], json!({})).await;

    let effort_req = cli.recv().await;
    assert_eq!(effort_req["request"]["subtype"], "apply_flag_settings");
    assert_eq!(effort_req["request"]["settings"]["effortLevel"], "xhigh");
    cli.ack_control(&effort_req["request_id"], json!({})).await;

    let off_req = cli.recv().await;
    assert_eq!(off_req["request"]["subtype"], "set_max_thinking_tokens");
    assert_eq!(off_req["request"]["max_thinking_tokens"], 0);
    cli.ack_control(&off_req["request_id"], json!({})).await;

    mock_slash_cli_turns(&mut cli).await;
}

async fn mock_slash_cli_turns(cli: &mut MockCliEndpoint) {
    let compact_in = cli.recv().await;
    assert_eq!(
        compact_in["message"]["content"][0]["text"],
        "/compact focus"
    );
    cli.send(&json!({
        "type": "system",
        "subtype": "compact_boundary",
        "session_id": "sess-ctrl"
    }))
    .await;
    send_ok_result(cli, 2, None).await;

    let clear_in = cli.recv().await;
    assert_eq!(clear_in["message"]["content"][0]["text"], "/clear");
    cli.send(&json!({
        "type": "conversation_reset",
        "new_conversation_id": "conv-new",
        "uuid": "rst-1",
        "session_id": "sess-ctrl"
    }))
    .await;
    send_ok_result(cli, 3, None).await;

    let review_in = cli.recv().await;
    assert_eq!(
        review_in["message"]["content"][0]["text"],
        "/review src/lib.rs"
    );
    send_ok_result(cli, 4, Some("No defects found in src/lib.rs.")).await;
}

async fn send_ok_result(cli: &mut MockCliEndpoint, num_turns: u32, result_text: Option<&str>) {
    let mut payload = json!({
        "type": "result",
        "subtype": "success",
        "is_error": false,
        "duration_ms": 5,
        "duration_api_ms": 2,
        "num_turns": num_turns,
        "session_id": "sess-ctrl",
        "total_cost_usd": 0.0
    });
    if let Some(txt) = result_text {
        payload["result"] = Value::String(txt.to_string());
    }
    cli.send(&payload).await;
}

async fn verify_modes_and_config_options(client: &AcpClient<RecordingClientHandler>, sid: &str) {
    for mode_id in ["read-only", "workspace-write", "agent-full-access"] {
        client
            .set_mode(SessionSetModeParams::new(sid, mode_id))
            .await
            .expect("set_mode");
    }

    for (opt_id, opt_val) in [
        ("model", "opus"),
        ("reasoning_effort", "xhigh"),
        ("reasoning_effort", "off"),
    ] {
        let updated = client
            .set_config_option(SessionSetConfigOptionParams::new(sid, opt_id, opt_val))
            .await
            .expect("set_config_option");
        assert!(updated.iter().any(|o| {
            o.id == opt_id
                && o.current_value == Some(ConfigOptionValue::Selected(opt_val.to_string()))
        }));
    }
}

#[tokio::test]
async fn mid_turn_cancel_modes_config_options_and_slash_commands() {
    let harness = TestHarness::start(false, 50);
    let _ = harness.client.initialize().await.expect("initialize");

    let (started_tx, started_rx) = async_channel::bounded(1);
    let cli_rx = harness.cli_rx.clone();
    let mock_task = tokio::spawn(async move {
        let cli = cli_rx.recv().await.unwrap();
        mock_cancel_modes_and_slash_cli(cli, started_tx).await;
    });

    let session = harness
        .client
        .new_session(SessionNewParams::new("/workspace"))
        .await
        .expect("new_session");
    let sid = session.session_id.clone();

    let prompt_client = harness.client.clone();
    let sid_clone = sid.clone();
    let prompt_task = tokio::spawn(async move {
        prompt_client
            .prompt(PromptParams::new(
                &sid_clone,
                vec![text_block("Long analysis")],
            ))
            .await
    });

    started_rx.recv().await.unwrap();
    harness
        .client
        .cancel(SessionCancelParams::new(&sid))
        .await
        .expect("cancel");
    let cancelled_res = prompt_task.await.unwrap().expect("cancelled prompt");
    assert_eq!(cancelled_res.stop_reason, StopReason::Cancelled);

    verify_modes_and_config_options(&harness.client, &sid).await;

    let _ = harness.handler.take_updates();
    for cmd in [
        "/status",
        "/logout",
        "/compact focus",
        "/clear",
        "/review src/lib.rs",
    ] {
        let res = harness
            .client
            .prompt(PromptParams::new(&sid, vec![text_block(cmd)]))
            .await
            .expect(cmd);
        assert_eq!(res.stop_reason, StopReason::EndTurn);
    }

    let slash_chunks: Vec<String> = harness
        .handler
        .take_updates()
        .into_iter()
        .filter_map(|n| match n.update {
            SessionUpdate::AgentMessageChunk(c) => match c.content {
                ContentBlock::Text(t) => Some(t.text),
                _ => None,
            },
            _ => None,
        })
        .collect();
    for expected_snippet in [
        "Mode: agent-full-access",
        "Signed out",
        "Conversation compacted.",
        "Conversation cleared.",
        "No defects found in src/lib.rs.",
    ] {
        assert!(
            slash_chunks.iter().any(|t| t.contains(expected_snippet)),
            "missing snippet {expected_snippet:?} in {slash_chunks:?}"
        );
    }

    mock_task.await.unwrap();
    harness.shutdown().await;
}

// =============================================================================
// 5. Session Lifecycle (`list` with pagination & CJK, `load` replay, `resume`, `close`, `delete`)
// =============================================================================

fn seed_persisted_transcripts(home: &Path, cwd: &Path, s_old: &str, s_cjk: &str, cjk_title: &str) {
    write_transcript_jsonl(
        &transcript_path(home, cwd, s_old),
        &[json!({
            "type": "user",
            "sessionId": s_old,
            "cwd": cwd,
            "timestamp": "2026-11-17T10:00:00Z",
            "message": { "role": "user", "content": "Initial setup" }
        })],
    );

    write_transcript_jsonl(
        &transcript_path(home, cwd, s_cjk),
        &[
            json!({
                "type": "user",
                "uuid": "u-1",
                "sessionId": s_cjk,
                "cwd": cwd,
                "timestamp": "2026-11-17T12:00:00Z",
                "message": { "role": "user", "content": [{"type": "text", "text": cjk_title}] }
            }),
            json!({
                "type": "assistant",
                "uuid": "a-1",
                "sessionId": s_cjk,
                "timestamp": "2026-11-17T12:00:02Z",
                "message": {
                    "id": "msg_hist_1",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [
                        { "type": "thinking", "thinking": "Checking auth.", "signature": "s" },
                        { "type": "text", "text": "Reading config." },
                        { "type": "tool_use", "id": "tu-h1", "name": "Read", "input": { "file_path": "/workspace/cjk-app/auth.rs" } }
                    ]
                }
            }),
            json!({
                "type": "user",
                "uuid": "u-tr-1",
                "sessionId": s_cjk,
                "cwd": cwd,
                "timestamp": "2026-11-17T12:00:03Z",
                "message": {
                    "role": "user",
                    "content": [{ "type": "tool_result", "tool_use_id": "tu-h1", "content": "pub fn check() {}" }]
                }
            }),
            json!({ "type": "ai-title", "sessionId": s_cjk, "aiTitle": cjk_title }),
        ],
    );
}

fn assert_replayed_cjk_history(replayed: &[SessionNotification], cjk_title: &str) {
    assert!(replayed.iter().any(|n| matches!(
        &n.update,
        SessionUpdate::UserMessageChunk(c)
            if matches!(&c.content, ContentBlock::Text(t) if t.text == cjk_title)
    )));
    assert!(
        replayed
            .iter()
            .any(|n| matches!(&n.update, SessionUpdate::AgentThoughtChunk(_)))
    );
    assert!(
        replayed
            .iter()
            .any(|n| matches!(&n.update, SessionUpdate::AgentMessageChunk(_)))
    );
    assert!(replayed.iter().any(|n| matches!(
        &n.update,
        SessionUpdate::ToolCall(tc) if tc.tool_call_id == "tu-h1" && tc.kind == Some(ToolKind::Read)
    )));
    assert!(replayed.iter().any(|n| matches!(
        &n.update,
        SessionUpdate::ToolCallUpdate(u) if u.tool_call_id == "tu-h1" && u.status == Some(ToolCallStatus::Completed)
    )));
}

#[tokio::test]
async fn session_list_pagination_cjk_load_replay_resume_close_and_delete() {
    let harness = TestHarness::start(false, 2);
    let _ = harness.client.initialize().await.expect("initialize");

    let cwd = PathBuf::from("/workspace/cjk-app");
    // Full CJK coverage: Chinese, Japanese, and Korean together.
    let cjk_title = "重构鉴权模块 / 認証モジュールの改修 / 인증 모듈 리팩터링";
    let s_old = "11111111-1111-4111-8111-111111111111";
    let s_cjk = "22222222-2222-4222-8222-222222222222";

    seed_persisted_transcripts(&harness.home.path, &cwd, s_old, s_cjk, cjk_title);

    let cli_rx = harness.cli_rx.clone();
    let mock_task = tokio::spawn(async move {
        let mut cli_live = cli_rx.recv().await.unwrap();
        assert!(cli_live.spawn_config.resume.is_none());
        let _ = cli_live.ack_initialize(vec![]).await;

        let mut cli_loaded = cli_rx.recv().await.unwrap();
        assert_eq!(cli_loaded.spawn_config.resume.as_deref(), Some(s_cjk));
        let _ = cli_loaded.ack_initialize(vec![]).await;

        let mut cli_resumed = cli_rx.recv().await.unwrap();
        assert_eq!(cli_resumed.spawn_config.resume.as_deref(), Some(s_old));
        let _ = cli_resumed.ack_initialize(vec![]).await;
    });

    let live = harness
        .client
        .new_session(
            SessionNewParams::new(&cwd)
                .additional_directories(vec![PathBuf::from("/workspace/shared")]),
        )
        .await
        .expect("new_session");

    let page1 = harness
        .client
        .list_sessions(SessionListParams::new().cwd(&cwd))
        .await
        .expect("list page 1");
    assert_eq!(page1.sessions.len(), 2);
    assert_eq!(page1.sessions[0].session_id, live.session_id);
    assert_eq!(
        page1.sessions[0].additional_directories,
        vec![PathBuf::from("/workspace/shared")]
    );
    assert_eq!(page1.sessions[1].session_id, s_cjk);
    assert_eq!(page1.sessions[1].title.as_deref(), Some(cjk_title));
    assert_eq!(page1.next_cursor.as_deref(), Some("2"));

    let page2 = harness
        .client
        .list_sessions(SessionListParams::new().cwd(&cwd).cursor("2"))
        .await
        .expect("list page 2");
    assert_eq!(page2.sessions.len(), 1);
    assert_eq!(page2.sessions[0].session_id, s_old);
    assert!(page2.next_cursor.is_none());

    let _ = harness.handler.take_updates();
    let loaded = harness
        .client
        .load_session(SessionLoadParams::new(s_cjk, &cwd))
        .await
        .expect("load_session");
    assert!(loaded.modes.is_some());
    assert_replayed_cjk_history(&harness.handler.take_updates(), cjk_title);

    let _ = harness.handler.take_updates();
    let _resumed = harness
        .client
        .resume_session(SessionResumeParams::new(s_old, &cwd))
        .await
        .expect("resume_session");
    assert!(!harness.handler.take_updates().iter().any(|n| matches!(
        &n.update,
        SessionUpdate::UserMessageChunk(_) | SessionUpdate::AgentMessageChunk(_)
    )));

    harness
        .client
        .close_session(SessionCloseParams::new(s_cjk))
        .await
        .expect("close_session");
    assert!(transcript_path(&harness.home.path, &cwd, s_cjk).exists());

    harness
        .client
        .delete_session(SessionDeleteParams::new(s_old))
        .await
        .expect("delete_session");
    assert!(!transcript_path(&harness.home.path, &cwd, s_old).exists());

    mock_task.await.unwrap();
    harness.shutdown().await;
}
