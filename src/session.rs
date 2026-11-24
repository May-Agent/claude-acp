//! [`ClaudeSession`]: SDK-backed [`aither_acp::Session`] implementation driving
//! prompt turns through [`Turn`]/[`SessionContext`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::SystemTime;

use aither_acp::domain::ConfigValueId;
use aither_acp::domain::{
    AvailableCommand, ConfigOption, ConfigValue, CurrentModeUpdate, Open, Patch, SessionInfo,
    SessionUpdate, StopReason, ToolCallContent, ToolCallStatus, ToolCallUpdate,
};
use aither_acp::rpc::RpcError;
use aither_acp::server::{
    MessageStream, PendingPrompt, PromptError, Session, SessionContext, Turn,
};
use aither_acp::{ConfigId, PlanId, SessionId, ToolCallId};
use chrono::{DateTime, SecondsFormat, Utc};
use claude_acp_sdk::{
    Client, CommandInfo, ContentBlock as SdkContentBlock, ContextUsageResponse, EffortLevel,
    InitializeResponse, McpStatusResponse, RateLimitEvent, RateLimitInfo, ResultMessage,
    ResultSubtype, SdkError, SessionEvent, StreamDelta, TextBlock, ToolProgressMessage,
    TranscriptSync, claude_codes::StopReason as ClaudeStopReason,
};
use serde_json::{Value, json};

use crate::agent::{AgentInner, NOTIFY_RATE_LIMIT};
use crate::auth::{AuthState, notify_auth_status_cx};
use crate::convert::{
    build_tool_call, build_tool_input_update, build_tool_result_update, convert_prompt_blocks,
    extract_todo_plan,
};
use crate::driver::DriverSlot;
use crate::error::{ClaudeAcpError, Result};
use crate::mode::{
    AcpMode, ReasoningSetting, build_available_commands, build_config_options,
    extract_model_options,
};
use crate::permission::PermissionMemory;

/// Plan id shared by all `TodoWrite`-derived plans.
const TODO_PLAN_ID: &str = "todos";

/// Mutable display and configuration metadata for a [`ClaudeSession`].
#[derive(Debug)]
pub struct SessionMetadata {
    pub(crate) model: String,
    pub(crate) reasoning: ReasoningSetting,
    pub(crate) available_models: Vec<aither_acp::domain::SelectOption>,
    pub(crate) cli_commands: Vec<CommandInfo>,
    pub(crate) title: Option<String>,
    pub(crate) updated_at: String,
    pub(crate) latest_rate_limit: Option<RateLimitEvent>,
    pub(crate) rate_limit_windows: BTreeMap<String, RateLimitInfo>,
}

/// State of an active SDK-backed ACP session.
#[derive(Debug)]
pub struct ClaudeSession {
    pub(crate) session_id: SessionId,
    pub(crate) cwd: PathBuf,
    pub(crate) additional_directories: Vec<PathBuf>,
    pub(crate) mcp_server_names: Vec<String>,
    pub(crate) client: Client,
    pub(crate) driver: DriverSlot,
    pub(crate) mode: Arc<Mutex<AcpMode>>,
    pub(crate) meta: Mutex<SessionMetadata>,
    pub(crate) permission_memory: Arc<PermissionMemory>,
    pub(crate) turn_cancelled: Arc<AtomicBool>,
    pub(crate) shutdown_started: AtomicBool,
    pub(crate) agent: Weak<AgentInner>,
}

impl ClaudeSession {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session_id: SessionId,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_server_names: Vec<String>,
        client: Client,
        driver: DriverSlot,
        mode: Arc<Mutex<AcpMode>>,
        initial_model: String,
        initial_reasoning: ReasoningSetting,
        init_response: InitializeResponse,
        permission_memory: Arc<PermissionMemory>,
        turn_cancelled: Arc<AtomicBool>,
        initial_title: Option<String>,
        agent: Weak<AgentInner>,
    ) -> Self {
        let available_models = extract_model_options(&init_response.models);
        Self {
            session_id,
            cwd,
            additional_directories,
            mcp_server_names,
            client,
            driver,
            mode,
            meta: Mutex::new(SessionMetadata {
                model: initial_model,
                reasoning: initial_reasoning,
                available_models,
                cli_commands: init_response.commands,
                title: initial_title,
                updated_at: now_rfc3339(),
                latest_rate_limit: None,
                rate_limit_windows: BTreeMap::new(),
            }),
            permission_memory,
            turn_cancelled,
            shutdown_started: AtomicBool::new(false),
            agent,
        }
    }

    pub(crate) fn current_mode(&self) -> AcpMode {
        *self.mode.lock().expect("session mode mutex poisoned")
    }

    pub(crate) fn set_current_mode(&self, mode: AcpMode) {
        *self.mode.lock().expect("session mode mutex poisoned") = mode;
    }

    pub(crate) fn current_config_options(&self) -> Vec<ConfigOption> {
        let mode = self.current_mode();
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        build_config_options(mode, &meta.model, &meta.available_models, &meta.reasoning)
    }

    pub(crate) fn current_commands(&self) -> Vec<AvailableCommand> {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        build_available_commands(&meta.cli_commands)
    }

    pub(crate) fn supports_cli_command(&self, name: &str) -> bool {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        meta.cli_commands
            .iter()
            .any(|cmd| cmd.name.trim_start_matches('/') == name)
    }

    pub(crate) fn session_info(&self) -> SessionInfo {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        let mut extra = BTreeMap::new();
        if !self.additional_directories.is_empty() {
            extra.insert(
                "additionalDirectories".to_string(),
                serde_json::json!(
                    self.additional_directories
                        .iter()
                        .map(|d| d.display().to_string())
                        .collect::<Vec<_>>()
                ),
            );
        }
        SessionInfo {
            session_id: self.session_id.clone(),
            cwd: self.cwd.clone(),
            title: meta.title.clone(),
            updated_at: Some(meta.updated_at.clone()),
            meta: None,
            extra,
        }
    }

    pub(crate) fn format_status_report(&self, auth_state: &AuthState) -> String {
        let mode = self.current_mode();
        let (model_str, effort_str) = {
            let meta = self.meta.lock().expect("session metadata mutex poisoned");
            (meta.model.clone(), meta.reasoning.current_value_id())
        };
        let auth = auth_state.status();
        let mcp_summary = if self.mcp_server_names.is_empty() {
            "none".to_string()
        } else {
            self.mcp_server_names.join(", ")
        };
        let auth_desc = auth.label.as_deref().map_or_else(
            || auth.kind.as_str().to_string(),
            |lbl| format!("{} ({lbl})", auth.kind.as_str()),
        );

        format!(
            "Session: {}\nDirectory: {}\nMode: {} ({})\nModel: {}\nReasoning effort: {}\nAuth: {}\nMCP servers: {}",
            self.session_id,
            self.cwd.display(),
            mode.as_str(),
            mode.to_permission_mode().as_str(),
            model_str,
            effort_str,
            auth_desc,
            mcp_summary,
        )
    }

    /// Query MCP server connection and tool status for this session.
    pub(crate) async fn mcp_status(&self) -> Result<McpStatusResponse> {
        self.driver
            .drive_until(self.client.mcp_status())
            .await
            .map_err(ClaudeAcpError::from)
    }

    /// Query structured context-window usage for this session.
    pub(crate) async fn context_usage(&self) -> Result<ContextUsageResponse> {
        self.driver
            .drive_until(self.client.context_usage())
            .await
            .map_err(ClaudeAcpError::from)
    }

    /// Snapshot the latest observed rate-limit event and per-window readings as JSON.
    pub(crate) fn rate_limits_json(&self) -> Value {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        json!({
            "sessionId": self.session_id.as_str(),
            "latest": meta.latest_rate_limit,
            "rateLimitInfo": meta.latest_rate_limit.as_ref().map(|ev| &ev.rate_limit_info),
            "windows": meta.rate_limit_windows,
        })
    }

    async fn push_transcript(&self) {
        let Some(agent) = self.agent.upgrade() else {
            return;
        };
        let store = match agent.resolve_transcript_store() {
            Ok(store) => store,
            Err(err) => {
                tracing::warn!(
                    session_id = %self.session_id,
                    %err,
                    "unable to resolve transcript home for transcript push"
                );
                return;
            }
        };
        if let Err(err) = agent
            .config
            .transcript_sync
            .push(self.session_id.as_str(), &self.cwd, store.home())
            .await
        {
            tracing::warn!(
                session_id = %self.session_id,
                cwd = %self.cwd.display(),
                home = %store.home().display(),
                %err,
                "transcript push failed"
            );
        }
    }

    /// Shut down the SDK client and drive its transport/process future to exit,
    /// then push local transcripts via the configured `TranscriptSync`.
    pub(crate) async fn shutdown(&self) {
        if self.shutdown_started.swap(true, Ordering::SeqCst) {
            return;
        }
        self.turn_cancelled.store(true, Ordering::SeqCst);
        self.client.close();
        self.driver.drive_to_completion().await;
        self.push_transcript().await;
    }

    /// Execute one prompt turn on this session, streaming updates through
    /// `turn`/`cx` until the CLI yields its terminal `ResultMessage` or the
    /// turn is cancelled.
    pub(crate) async fn run_turn(
        &self,
        turn: &Turn,
        cx: &SessionContext,
        auth_state: &Mutex<AuthState>,
        blocks: Vec<SdkContentBlock>,
        fallback_title_text: Option<&str>,
    ) -> Result<StopReason> {
        self.turn_cancelled.store(false, Ordering::SeqCst);

        let title_update = {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            meta.updated_at = now_rfc3339();
            if meta.title.is_none()
                && let Some(raw_title) = fallback_title_text
                && let Some(clean_title) = derive_prompt_title(raw_title)
            {
                meta.title = Some(clean_title.clone());
                Some((clean_title, meta.updated_at.clone()))
            } else {
                None
            }
        };
        if let Some((clean_title, updated_at)) = title_update {
            cx.set_session_info(Patch::Set(clean_title), Patch::Set(updated_at))
                .await
                .map_err(|e| ClaudeAcpError::Server(e.into()))?;
        }

        let stop_reason = Box::pin(self.driver.drive_until(async {
            self.client.send_user_blocks(blocks, None).await?;
            let mut streamer = TurnStreamer::new();

            loop {
                let next_ev = self.client.next_event();
                futures_lite::pin!(next_ev);
                let next = futures_lite::future::or(
                    async { Next::Event(Box::new(next_ev.await)) },
                    async {
                        turn.cancelled().await;
                        Next::Cancelled
                    },
                )
                .await;

                match next {
                    Next::Cancelled => {
                        self.turn_cancelled.store(true, Ordering::SeqCst);
                        if let Err(err) = self.client.interrupt().await {
                            tracing::debug!(%err, "interrupt after cancel ended with error");
                        }
                        // Keep draining until the CLI reports a result or
                        // the channel closes; either resolves the turn.
                    }
                    Next::Event(evt) => match *evt {
                        Ok(Some(event)) => {
                            if let Some(stop_reason) = Box::pin(self.handle_session_event(
                                turn,
                                cx,
                                auth_state,
                                &mut streamer,
                                Box::new(event),
                            ))
                            .await?
                            {
                                return Ok(stop_reason);
                            }
                        }
                        Ok(None) => {
                            if self.turn_cancelled.load(Ordering::SeqCst) {
                                return Ok(StopReason::Cancelled);
                            }
                            return Err(ClaudeAcpError::Sdk(SdkError::ChannelClosed));
                        }
                        Err(err) => {
                            if self.turn_cancelled.load(Ordering::SeqCst) {
                                return Ok(StopReason::Cancelled);
                            }
                            return Err(ClaudeAcpError::Sdk(err));
                        }
                    },
                }
            }
        }))
        .await?;

        self.push_transcript().await;
        Ok(stop_reason)
    }

    async fn handle_session_event(
        &self,
        turn: &Turn,
        cx: &SessionContext,
        auth_state: &Mutex<AuthState>,
        streamer: &mut TurnStreamer,
        event: Box<SessionEvent>,
    ) -> Result<Option<StopReason>> {
        match *event {
            SessionEvent::StreamEvent(msg) => {
                if msg.parent_tool_use_id.is_none() {
                    self.handle_stream_delta(turn, streamer, StreamDelta::from_stream_event(&msg))
                        .await?;
                }
            }
            SessionEvent::Assistant(assistant) => {
                if assistant.parent_tool_use_id.is_none() && assistant.agent_id.is_none() {
                    self.handle_assistant_message(turn, streamer, &assistant)
                        .await?;
                }
            }
            SessionEvent::ToolProgress(prog) => {
                self.handle_tool_progress(turn, streamer, &prog).await?;
            }
            SessionEvent::User(user) => {
                if user.parent_tool_use_id.is_none() && user.agent_id.is_none() {
                    for block in &user.message.content {
                        if let SdkContentBlock::ToolResult(tr) = block {
                            if streamer.todo_ids.contains(&tr.tool_use_id) {
                                continue;
                            }
                            let diffs = streamer.tool_diffs.get(&tr.tool_use_id).map(Vec::as_slice);
                            let upd =
                                build_tool_result_update(tr, user.tool_use_result.as_ref(), diffs);
                            turn.update(SessionUpdate::ToolCallUpdate(upd))
                                .await
                                .map_err(into_server_err)?;
                        }
                    }
                }
            }
            SessionEvent::AuthStatus(auth_msg) => {
                let updated = auth_state
                    .lock()
                    .expect("auth state mutex poisoned")
                    .observe_cli_auth_status(&auth_msg);
                notify_auth_status_cx(cx, &updated).await?;
            }
            SessionEvent::System(sys) => {
                self.handle_system_message(cx, streamer, &sys).await?;
            }
            SessionEvent::ConversationReset(_) => {
                self.permission_memory.clear();
                streamer.notice = TurnNotice::Reset;
            }
            SessionEvent::Other(other) => {
                self.handle_other_output(cx, &other).await?;
            }
            SessionEvent::RateLimit(ev) => {
                self.handle_rate_limit(cx, ev).await?;
            }
            SessionEvent::Result(res) => {
                let final_stop = self.finish_turn(turn, streamer, &res).await?;
                return Ok(Some(final_stop));
            }
            SessionEvent::ApiError(_)
            | SessionEvent::CommandLifecycle(_)
            | SessionEvent::ToolUseSummary(_)
            | SessionEvent::PromptSuggestion(_) => {}
        }
        Ok(None)
    }

    async fn handle_stream_delta(
        &self,
        turn: &Turn,
        streamer: &mut TurnStreamer,
        delta: Option<StreamDelta>,
    ) -> Result<()> {
        let Some(delta) = delta else {
            return Ok(());
        };
        match delta {
            StreamDelta::MessageStart { message_id } => {
                streamer.current_message_id = message_id;
                streamer.msg_stream = None;
                streamer.thought_stream = None;
                streamer.last_delta_stop_reason = None;
            }
            StreamDelta::Thinking { thinking, .. } => {
                if !thinking.is_empty() {
                    streamer.streamed_thinking_bytes += thinking.len();
                    streamer
                        .thought_stream(turn)
                        .text(&thinking)
                        .await
                        .map_err(into_server_err)?;
                }
            }
            StreamDelta::Text { text, .. } => {
                if !text.is_empty() {
                    streamer.streamed_text_bytes += text.len();
                    streamer.text_state = TextProgress::Emitted;
                    streamer
                        .msg_stream(turn)
                        .text(&text)
                        .await
                        .map_err(into_server_err)?;
                }
            }
            StreamDelta::ToolUseStart { index, id, name } => {
                let is_todo = name == "TodoWrite";
                if !is_todo && streamer.announced_tools.insert(id.clone()) {
                    let call = build_tool_call(&id, &name, &Value::Null, ToolCallStatus::Pending);
                    turn.update(SessionUpdate::ToolCallUpdate(call))
                        .await
                        .map_err(into_server_err)?;
                }
                streamer.stream_tools.insert(
                    index,
                    PartialStreamTool {
                        id,
                        name,
                        json_buf: String::new(),
                    },
                );
            }
            StreamDelta::ToolInputJson {
                index,
                partial_json,
            } => {
                if let Some(entry) = streamer.stream_tools.get_mut(&index) {
                    entry.json_buf.push_str(&partial_json);
                }
            }
            StreamDelta::ContentBlockStop { index } => {
                if let Some(partial) = streamer.stream_tools.remove(&index) {
                    self.flush_partial_tool(turn, streamer, partial).await?;
                }
            }
            StreamDelta::MessageDelta { stop_reason } => {
                streamer.last_delta_stop_reason = stop_reason;
            }
            StreamDelta::Signature { .. } | StreamDelta::MessageStop => {}
        }
        Ok(())
    }

    async fn flush_partial_tool(
        &self,
        turn: &Turn,
        streamer: &mut TurnStreamer,
        partial: PartialStreamTool,
    ) -> Result<()> {
        if partial.json_buf.trim().is_empty() {
            return Ok(());
        }
        let Ok(parsed_input) = serde_json::from_str::<Value>(&partial.json_buf) else {
            return Ok(());
        };

        if let Some(entries) = extract_todo_plan(&partial.name, &parsed_input) {
            if streamer.todo_ids.insert(partial.id) {
                turn.plan(PlanId::from_string(TODO_PLAN_ID), entries)
                    .await
                    .map_err(into_server_err)?;
            }
        } else if streamer.enriched_tools.insert(partial.id.clone()) {
            let upd = build_tool_input_update(
                &partial.id,
                &partial.name,
                &parsed_input,
                ToolCallStatus::InProgress,
            );
            if let Patch::Set(diffs) = &upd.content {
                streamer.tool_diffs.insert(partial.id, diffs.clone());
            }
            turn.update(SessionUpdate::ToolCallUpdate(upd))
                .await
                .map_err(into_server_err)?;
        }
        Ok(())
    }

    async fn handle_assistant_message(
        &self,
        turn: &Turn,
        streamer: &mut TurnStreamer,
        assistant: &claude_acp_sdk::AssistantMessage,
    ) -> Result<()> {
        streamer
            .last_assistant_stop_reason
            .clone_from(&assistant.message.stop_reason);
        streamer.last_assistant_refused = assistant.refused_message_id.is_some();

        // 1. De-duplicate Thinking blocks against prior `thinking_delta` chunks.
        let mut full_thinking = String::new();
        for block in &assistant.message.content {
            if let SdkContentBlock::Thinking(tb) = block {
                full_thinking.push_str(&tb.thinking);
            }
        }
        if !full_thinking.is_empty() {
            if streamer.streamed_thinking_bytes == 0 {
                streamer
                    .thought_stream(turn)
                    .text(&full_thinking)
                    .await
                    .map_err(into_server_err)?;
            } else if full_thinking.len() > streamer.streamed_thinking_bytes
                && full_thinking.is_char_boundary(streamer.streamed_thinking_bytes)
            {
                let suffix = &full_thinking[streamer.streamed_thinking_bytes..];
                streamer
                    .thought_stream(turn)
                    .text(suffix)
                    .await
                    .map_err(into_server_err)?;
            }
        }

        // 2. De-duplicate Text blocks against prior `text_delta` chunks.
        let mut full_text = String::new();
        for block in &assistant.message.content {
            if let SdkContentBlock::Text(tb) = block {
                full_text.push_str(&tb.text);
            }
        }
        if !full_text.is_empty() {
            if streamer.streamed_text_bytes == 0 {
                streamer.text_state = TextProgress::Emitted;
                streamer
                    .msg_stream(turn)
                    .text(&full_text)
                    .await
                    .map_err(into_server_err)?;
            } else if full_text.len() > streamer.streamed_text_bytes
                && full_text.is_char_boundary(streamer.streamed_text_bytes)
            {
                let suffix = &full_text[streamer.streamed_text_bytes..];
                streamer.text_state = TextProgress::Emitted;
                streamer
                    .msg_stream(turn)
                    .text(suffix)
                    .await
                    .map_err(into_server_err)?;
            }
        }

        // 3. Emit or enrich tool calls & TodoWrite plans.
        for block in &assistant.message.content {
            match block {
                SdkContentBlock::ToolUse(tu) => {
                    self.emit_or_enrich_tool(turn, streamer, &tu.id, &tu.name, &tu.input)
                        .await?;
                }
                SdkContentBlock::McpToolUse(mcp) => {
                    self.emit_or_enrich_tool(turn, streamer, &mcp.id, &mcp.name, &mcp.input)
                        .await?;
                }
                SdkContentBlock::ServerToolUse(srv) => {
                    self.emit_or_enrich_tool(turn, streamer, &srv.id, &srv.name, &srv.input)
                        .await?;
                }
                _ => {}
            }
        }

        // Reset per-assistant-step streaming counters for multi-turn tool loops.
        streamer.streamed_thinking_bytes = 0;
        streamer.streamed_text_bytes = 0;
        streamer.stream_tools.clear();
        streamer.current_message_id = None;
        streamer.msg_stream = None;
        streamer.thought_stream = None;
        Ok(())
    }

    async fn emit_or_enrich_tool(
        &self,
        turn: &Turn,
        streamer: &mut TurnStreamer,
        id: &str,
        name: &str,
        input: &Value,
    ) -> Result<()> {
        if let Some(entries) = extract_todo_plan(name, input) {
            if streamer.todo_ids.insert(id.to_string()) {
                turn.plan(PlanId::from_string(TODO_PLAN_ID), entries)
                    .await
                    .map_err(into_server_err)?;
            }
            return Ok(());
        }

        if streamer.announced_tools.insert(id.to_string()) {
            streamer.enriched_tools.insert(id.to_string());
            let call = build_tool_call(id, name, input, ToolCallStatus::InProgress);
            if let Patch::Set(diffs) = &call.content
                && !diffs.is_empty()
            {
                streamer.tool_diffs.insert(id.to_string(), diffs.clone());
            }
            turn.update(SessionUpdate::ToolCallUpdate(call))
                .await
                .map_err(into_server_err)?;
        } else if streamer.enriched_tools.insert(id.to_string()) {
            let upd = build_tool_input_update(id, name, input, ToolCallStatus::InProgress);
            if let Patch::Set(diffs) = &upd.content {
                streamer.tool_diffs.insert(id.to_string(), diffs.clone());
            }
            turn.update(SessionUpdate::ToolCallUpdate(upd))
                .await
                .map_err(into_server_err)?;
        }
        Ok(())
    }

    async fn handle_tool_progress(
        &self,
        turn: &Turn,
        streamer: &TurnStreamer,
        prog: &ToolProgressMessage,
    ) -> Result<()> {
        if prog.heartbeat == Some(true)
            || streamer.todo_ids.contains(&prog.tool_use_id)
            || !streamer.announced_tools.contains(&prog.tool_use_id)
        {
            return Ok(());
        }
        let upd = ToolCallUpdate {
            tool_call_id: ToolCallId::from_string(prog.tool_use_id.clone()),
            status: Patch::Set(Open::Known(ToolCallStatus::InProgress)),
            ..ToolCallUpdate::default()
        };
        turn.update(SessionUpdate::ToolCallUpdate(upd))
            .await
            .map_err(into_server_err)
    }

    async fn handle_system_message(
        &self,
        cx: &SessionContext,
        streamer: &mut TurnStreamer,
        sys: &claude_acp_sdk::SystemMessage,
    ) -> Result<()> {
        if sys.is_compact_boundary() {
            streamer.notice = TurnNotice::Compacted;
            return Ok(());
        }

        if let Some(init) = sys.as_init() {
            let mut mode_changed = None;
            let mut config_changed = false;

            if let Some(cli_perm_mode) = &init.permission_mode {
                let mapped = AcpMode::from_permission_mode(cli_perm_mode);
                if mapped != self.current_mode() {
                    self.set_current_mode(mapped);
                    mode_changed = Some(mapped);
                    config_changed = true;
                }
            }

            config_changed |=
                self.reconcile_init_metadata(init.model.as_deref(), init.effort.as_deref());

            if let Some(new_mode) = mode_changed {
                cx.update(SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
                    current_mode_id: ConfigValueId::from_string(new_mode.as_str()),
                    meta: None,
                    extra: BTreeMap::new(),
                }))
                .await
                .map_err(into_server_err)?;
            }
            if config_changed {
                cx.set_config_options(self.current_config_options())
                    .await
                    .map_err(into_server_err)?;
            }
            return Ok(());
        }

        if let Some(new_title) = sys
            .data
            .get("title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            self.apply_title_update(cx, new_title.to_string()).await?;
        }
        Ok(())
    }

    fn reconcile_init_metadata(&self, model: Option<&str>, effort: Option<&str>) -> bool {
        let mut changed = false;
        let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
        if let Some(m) = model.filter(|s| !s.is_empty())
            && meta.model != m
        {
            meta.model = m.to_string();
            changed = true;
        }
        if let Some(eff_str) = effort.filter(|s| !s.is_empty()) {
            let eff = ReasoningSetting::Effort(EffortLevel::from(eff_str));
            if meta.reasoning != eff {
                meta.reasoning = eff;
                changed = true;
            }
        }
        drop(meta);
        changed
    }

    async fn handle_other_output(
        &self,
        cx: &SessionContext,
        output: &claude_acp_sdk::ClaudeOutput,
    ) -> Result<()> {
        let maybe_title = match output {
            claude_acp_sdk::ClaudeOutput::AiTitle(_) | claude_acp_sdk::ClaudeOutput::Summary(_) => {
                serde_json::to_value(output).ok().and_then(|val| {
                    val.get("aiTitle")
                        .or_else(|| val.get("summary"))
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(ToOwned::to_owned)
                })
            }
            _ => None,
        };
        if let Some(title) = maybe_title {
            self.apply_title_update(cx, title).await?;
        }
        Ok(())
    }

    async fn apply_title_update(&self, cx: &SessionContext, title: String) -> Result<()> {
        let updated_at = {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            meta.title = Some(title.clone());
            meta.updated_at = now_rfc3339();
            meta.updated_at.clone()
        };
        cx.set_session_info(Patch::Set(title), Patch::Set(updated_at))
            .await
            .map_err(into_server_err)
    }

    async fn handle_rate_limit(&self, cx: &SessionContext, ev: RateLimitEvent) -> Result<()> {
        let payload = json!({
            "sessionId": self.session_id.as_str(),
            "rateLimitInfo": &ev.rate_limit_info,
            "event": &ev,
        });
        {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            if let Some(window) = &ev.rate_limit_info.rate_limit_type {
                meta.rate_limit_windows
                    .insert(window.as_str().to_string(), ev.rate_limit_info.clone());
            }
            meta.latest_rate_limit = Some(ev);
        }
        cx.notify(NOTIFY_RATE_LIMIT, Some(payload))
            .await
            .map_err(into_server_err)
    }

    async fn finish_turn(
        &self,
        turn: &Turn,
        streamer: &mut TurnStreamer,
        res: &ResultMessage,
    ) -> Result<StopReason> {
        let remaining: Vec<PartialStreamTool> = std::mem::take(&mut streamer.stream_tools)
            .into_values()
            .collect();
        for partial in remaining {
            self.flush_partial_tool(turn, streamer, partial).await?;
        }

        {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            meta.updated_at = now_rfc3339();
        }

        if self.turn_cancelled.load(Ordering::SeqCst) {
            return Ok(StopReason::Cancelled);
        }

        if streamer.text_state == TextProgress::Empty {
            let fallback_text = res
                .result
                .as_deref()
                .filter(|s| !s.trim().is_empty())
                .map(ToOwned::to_owned)
                .or_else(|| match streamer.notice {
                    TurnNotice::Compacted => Some("Conversation compacted.".to_string()),
                    TurnNotice::Reset => Some("Conversation cleared.".to_string()),
                    TurnNotice::None => None,
                });

            if let Some(text) = fallback_text {
                turn.message().text(&text).await.map_err(into_server_err)?;
            }
        }

        Ok(map_stop_reason(
            res,
            streamer.last_assistant_stop_reason.as_ref(),
            streamer.last_delta_stop_reason.as_deref(),
            streamer.last_assistant_refused,
        ))
    }
}

impl Session for ClaudeSession {
    fn config_options(&self) -> Vec<ConfigOption> {
        self.current_config_options()
    }

    fn available_commands(&self) -> Vec<AvailableCommand> {
        self.current_commands()
    }

    async fn set_config(
        &self,
        id: &ConfigId,
        value: ConfigValue,
        _cx: &SessionContext,
    ) -> std::result::Result<Vec<ConfigOption>, RpcError> {
        let raw_value = match &value {
            ConfigValue::Id(v) => v.value.as_str().to_string(),
            other => {
                return Err(RpcError::invalid_params(format!(
                    "config option `{}` expects a string select value, got {other:?}",
                    id.as_str()
                )));
            }
        };

        let run = async {
            match id.as_str() {
                "mode" => {
                    let target_mode = AcpMode::parse(&raw_value)?;
                    self.driver
                        .drive_until(
                            self.client
                                .set_permission_mode(target_mode.to_permission_mode()),
                        )
                        .await?;
                    self.set_current_mode(target_mode);
                }
                "model" => {
                    if raw_value.trim().is_empty() {
                        return Err(ClaudeAcpError::InvalidParams(
                            "`model` value must not be empty".to_string(),
                        ));
                    }
                    let model_arg = (raw_value != "default").then_some(raw_value.as_str());
                    self.driver
                        .drive_until(self.client.set_model(model_arg))
                        .await?;
                    self.meta
                        .lock()
                        .expect("session metadata mutex poisoned")
                        .model
                        .clone_from(&raw_value);
                }
                "reasoning_effort" => {
                    apply_reasoning_option(self, &raw_value).await?;
                }
                other => {
                    return Err(ClaudeAcpError::InvalidParams(format!(
                        "unknown config option `{other}`; expected `mode`, `model`, or `reasoning_effort`"
                    )));
                }
            }
            Ok::<(), ClaudeAcpError>(())
        };
        run.await.map_err(ClaudeAcpError::into_rpc_error)?;
        Ok(self.current_config_options())
    }

    async fn prompt(
        &self,
        prompt: PendingPrompt,
        cx: &SessionContext,
    ) -> std::result::Result<StopReason, PromptError> {
        let agent = self.agent.upgrade();
        let auth_state = agent
            .as_ref()
            .map(|inner| &inner.auth)
            .ok_or_else(|| PromptError::Failed(RpcError::internal_error("agent dropped")))?;

        let slash = parse_slash_command(prompt.content());
        let blocks = if slash.is_none() {
            convert_prompt_blocks(prompt.content())
                .map_err(|e| PromptError::Failed(e.into_rpc_error()))?
        } else {
            Vec::new()
        };
        let first_text = first_prompt_text(prompt.content()).map(ToOwned::to_owned);

        let turn = prompt
            .accept()
            .await
            .map_err(|rejected| PromptError::Failed(rejected.into_inner()))?;

        let inner = || async {
            match slash {
                Some(cmd) => {
                    self.dispatch_slash_command(&turn, cx, auth_state, cmd)
                        .await
                }
                None => {
                    self.run_turn(&turn, cx, auth_state, blocks, first_text.as_deref())
                        .await
                }
            }
        };
        inner().await.map_err(|e| {
            if matches!(e, ClaudeAcpError::Sdk(SdkError::ChannelClosed))
                && self.turn_cancelled.load(Ordering::SeqCst)
            {
                PromptError::Cancelled
            } else {
                PromptError::Failed(e.into_rpc_error())
            }
        })
    }

    async fn close(&self) {
        self.shutdown().await;
        if let Some(inner) = self.agent.upgrade() {
            inner
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .remove(self.session_id.as_str());
        }
    }
}

impl ClaudeSession {
    async fn dispatch_slash_command(
        &self,
        turn: &Turn,
        cx: &SessionContext,
        auth_state: &Mutex<AuthState>,
        cmd: SlashCommand,
    ) -> Result<StopReason> {
        match cmd.name.as_str() {
            "status" => {
                let report = {
                    let auth = auth_state.lock().expect("auth mutex poisoned");
                    self.format_status_report(&auth)
                };
                turn.message()
                    .text(&report)
                    .await
                    .map_err(into_server_err)?;
                Ok(StopReason::EndTurn)
            }
            "logout" => {
                let status = {
                    let mut auth = auth_state.lock().expect("auth mutex poisoned");
                    auth.logout()
                };
                notify_auth_status_cx(cx, &status).await?;
                turn.message()
                    .text("Signed out of Claude session credentials.")
                    .await
                    .map_err(into_server_err)?;
                Ok(StopReason::EndTurn)
            }
            "clear" | "compact" | "review" => {
                let text = slash_command_text(self, &cmd);
                let blocks = vec![SdkContentBlock::Text(TextBlock {
                    text,
                    citations: Vec::new(),
                })];
                self.run_turn(turn, cx, auth_state, blocks, None).await
            }
            _ => unreachable!("only intercepted commands are matched"),
        }
    }
}

fn slash_command_text(session: &ClaudeSession, cmd: &SlashCommand) -> String {
    match cmd.name.as_str() {
        "clear" => "/clear".to_string(),
        "compact" => {
            if cmd.args.is_empty() {
                "/compact".to_string()
            } else {
                format!("/compact {}", cmd.args)
            }
        }
        "review" => {
            if session.supports_cli_command("review") {
                if cmd.args.is_empty() {
                    "/review".to_string()
                } else {
                    format!("/review {}", cmd.args)
                }
            } else if cmd.args.is_empty() {
                "Review the current workspace changes (git status and diff) for bugs, regressions, and missing test coverage.".to_string()
            } else {
                format!(
                    "Review `{}` for bugs, regressions, and missing test coverage.",
                    cmd.args
                )
            }
        }
        _ => unreachable!(),
    }
}

/// Shared handle to a live [`ClaudeSession`]: the [`SessionHost`](aither_acp::SessionHost)
/// `Session` type the framework dispatches to while the agent keeps a registry
/// of open sessions.
#[derive(Clone)]
pub struct ClaudeSessionHandle(pub(crate) Arc<ClaudeSession>);

impl std::fmt::Debug for ClaudeSessionHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeSessionHandle")
            .field("session_id", &self.0.session_id)
            .finish_non_exhaustive()
    }
}

impl std::ops::Deref for ClaudeSessionHandle {
    type Target = ClaudeSession;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Session for ClaudeSessionHandle {
    fn config_options(&self) -> Vec<ConfigOption> {
        self.0.current_config_options()
    }

    fn available_commands(&self) -> Vec<AvailableCommand> {
        self.0.current_commands()
    }

    async fn set_config(
        &self,
        id: &ConfigId,
        value: ConfigValue,
        cx: &SessionContext,
    ) -> std::result::Result<Vec<ConfigOption>, RpcError> {
        Session::set_config(&*self.0, id, value, cx).await
    }

    async fn prompt(
        &self,
        prompt: PendingPrompt,
        cx: &SessionContext,
    ) -> std::result::Result<StopReason, PromptError> {
        Session::prompt(&*self.0, prompt, cx).await
    }

    async fn close(&self) {
        Session::close(&*self.0).await;
    }
}

/// Next item yielded while racing the SDK event stream against
/// `Turn::cancelled` inside `run_turn`.
enum Next {
    Event(Box<std::result::Result<Option<SessionEvent>, SdkError>>),
    Cancelled,
}

struct SlashCommand {
    name: String,
    args: String,
}

fn parse_slash_command(prompt: &[aither_acp::ContentBlock]) -> Option<SlashCommand> {
    if prompt.len() != 1 {
        return None;
    }
    let aither_acp::ContentBlock::Text(text_block) = &prompt[0] else {
        return None;
    };
    let trimmed = text_block.text.trim();
    let without_slash = trimmed.strip_prefix('/')?;
    let (name, args) = match without_slash.split_once(char::is_whitespace) {
        Some((n, rest)) => (n, rest.trim()),
        None => (without_slash, ""),
    };
    match name {
        "compact" | "clear" | "status" | "logout" | "review" => Some(SlashCommand {
            name: name.to_string(),
            args: args.to_string(),
        }),
        _ => None,
    }
}

fn first_prompt_text(prompt: &[aither_acp::ContentBlock]) -> Option<&str> {
    prompt.iter().find_map(|block| match block {
        aither_acp::ContentBlock::Text(t) if !t.text.trim().is_empty() => Some(t.text.as_str()),
        _ => None,
    })
}

async fn apply_reasoning_option(session: &ClaudeSession, value: &str) -> Result<()> {
    let next_setting = ReasoningSetting::parse(value)?;
    let prev_was_custom_cap = {
        let meta = session
            .meta
            .lock()
            .expect("session metadata mutex poisoned");
        matches!(
            meta.reasoning,
            ReasoningSetting::Off | ReasoningSetting::Tokens(_)
        )
    };

    match &next_setting {
        ReasoningSetting::Off => {
            session
                .driver
                .drive_until(session.client.set_max_thinking_tokens(Some(0)))
                .await?;
        }
        ReasoningSetting::Tokens(tokens) => {
            session
                .driver
                .drive_until(session.client.set_max_thinking_tokens(Some(*tokens)))
                .await?;
        }
        ReasoningSetting::Effort(level) => {
            if prev_was_custom_cap {
                session
                    .driver
                    .drive_until(session.client.set_max_thinking_tokens(None))
                    .await?;
            }
            session
                .driver
                .drive_until(session.client.set_effort(level.clone()))
                .await?;
        }
    }

    session
        .meta
        .lock()
        .expect("session metadata mutex poisoned")
        .reasoning = next_setting;
    Ok(())
}

#[derive(Debug)]
struct PartialStreamTool {
    id: String,
    name: String,
    json_buf: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum TextProgress {
    #[default]
    Empty,
    Emitted,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum TurnNotice {
    #[default]
    None,
    Compacted,
    Reset,
}

struct TurnStreamer {
    current_message_id: Option<String>,
    msg_stream: Option<MessageStream>,
    thought_stream: Option<MessageStream>,
    streamed_thinking_bytes: usize,
    streamed_text_bytes: usize,
    text_state: TextProgress,
    notice: TurnNotice,
    stream_tools: HashMap<u64, PartialStreamTool>,
    announced_tools: HashSet<String>,
    enriched_tools: HashSet<String>,
    todo_ids: HashSet<String>,
    tool_diffs: HashMap<String, Vec<ToolCallContent>>,
    last_assistant_stop_reason: Option<ClaudeStopReason>,
    last_delta_stop_reason: Option<String>,
    last_assistant_refused: bool,
}

impl std::fmt::Debug for TurnStreamer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnStreamer").finish_non_exhaustive()
    }
}

impl TurnStreamer {
    fn new() -> Self {
        Self {
            current_message_id: None,
            msg_stream: None,
            thought_stream: None,
            streamed_thinking_bytes: 0,
            streamed_text_bytes: 0,
            text_state: TextProgress::Empty,
            notice: TurnNotice::None,
            stream_tools: HashMap::new(),
            announced_tools: HashSet::new(),
            enriched_tools: HashSet::new(),
            todo_ids: HashSet::new(),
            tool_diffs: HashMap::new(),
            last_assistant_stop_reason: None,
            last_delta_stop_reason: None,
            last_assistant_refused: false,
        }
    }

    fn msg_stream(&mut self, turn: &Turn) -> &MessageStream {
        self.msg_stream.get_or_insert_with(|| turn.message())
    }

    fn thought_stream(&mut self, turn: &Turn) -> &MessageStream {
        self.thought_stream.get_or_insert_with(|| turn.thought())
    }
}

fn map_stop_reason(
    res: &ResultMessage,
    assistant_stop: Option<&ClaudeStopReason>,
    delta_stop: Option<&str>,
    refused: bool,
) -> StopReason {
    if refused || res.stop_reason.as_deref() == Some("refusal") || delta_stop == Some("refusal") {
        return StopReason::Refusal;
    }

    if matches!(res.subtype, ResultSubtype::ErrorMaxTurns) {
        return StopReason::MaxTurnRequests;
    }

    if res.stop_reason.as_deref() == Some("max_tokens")
        || delta_stop == Some("max_tokens")
        || matches!(assistant_stop, Some(ClaudeStopReason::MaxTokens))
    {
        return StopReason::MaxTokens;
    }

    if res.is_error
        || matches!(
            res.subtype,
            ResultSubtype::ErrorDuringExecution
                | ResultSubtype::ErrorMaxBudgetUsd
                | ResultSubtype::ErrorMaxStructuredOutputRetries
        )
    {
        return StopReason::Error;
    }

    StopReason::EndTurn
}

fn into_server_err(err: RpcError) -> ClaudeAcpError {
    ClaudeAcpError::Server(err.into())
}

/// Current wall-clock UTC timestamp rendered as RFC 3339 (`...Z`).
#[must_use]
pub fn now_rfc3339() -> String {
    DateTime::<Utc>::from(SystemTime::now()).to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn derive_prompt_title(raw: &str) -> Option<String> {
    const MAX_TITLE_CHARS: usize = 80;
    let collapsed: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.is_empty() {
        return None;
    }
    let mut chars = collapsed.chars();
    let head: String = chars.by_ref().take(MAX_TITLE_CHARS).collect();
    if chars.next().is_some() {
        Some(format!("{head}…"))
    } else {
        Some(head)
    }
}
