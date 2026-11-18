//! Live session state and streaming `SessionEvent` -> `SessionUpdate` turn driver.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use aither_acp::{
    AvailableCommandsUpdate, ClientPeer, ConfigOption, ConfigOptionUpdate, ConfigSelectOption,
    CurrentModeUpdate, PromptResult, SessionInfo, SessionInfoUpdate, SessionModeState,
    SessionNotification, SessionUpdate, StopReason, ToolCallContent, ToolCallStatus,
};
use chrono::{DateTime, SecondsFormat, Utc};
use claude_acp_sdk::{
    ClaudeOutput, Client, CommandInfo, ContentBlock as SdkContentBlock, EffortLevel,
    InitializeResponse, ResultMessage, ResultSubtype, SdkError, SessionEvent, StreamDelta,
    ToolProgressMessage, claude_codes::StopReason as ClaudeStopReason,
};
use serde_json::Value;

use crate::auth::{AuthState, notify_auth_status};
use crate::convert::{
    acp_text_chunk, build_tool_call, build_tool_input_update, build_tool_result_update,
    extract_todo_plan,
};
use crate::driver::DriverSlot;
use crate::error::{ClaudeAcpError, Result};
use crate::mode::{
    AcpMode, ReasoningSetting, build_available_commands, build_config_options,
    extract_model_options,
};
use crate::permission::PermissionMemory;

/// Mutable display and configuration metadata for a [`LiveSession`].
#[derive(Debug)]
pub struct SessionMetadata {
    pub(crate) model: String,
    pub(crate) reasoning: ReasoningSetting,
    pub(crate) available_models: Vec<ConfigSelectOption>,
    pub(crate) cli_commands: Vec<CommandInfo>,
    pub(crate) title: Option<String>,
    pub(crate) updated_at: String,
}

/// State of an active SDK-backed ACP session.
#[derive(Debug)]
pub struct LiveSession {
    pub(crate) session_id: String,
    pub(crate) cwd: PathBuf,
    pub(crate) additional_directories: Vec<PathBuf>,
    pub(crate) mcp_server_names: Vec<String>,
    pub(crate) client: Client,
    pub(crate) driver: DriverSlot,
    pub(crate) mode: Arc<Mutex<AcpMode>>,
    pub(crate) meta: Mutex<SessionMetadata>,
    pub(crate) permission_memory: Arc<PermissionMemory>,
    pub(crate) turn_cancelled: Arc<AtomicBool>,
    pub(crate) turn_gate: async_lock::Mutex<()>,
}

impl LiveSession {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        session_id: String,
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
            }),
            permission_memory,
            turn_cancelled,
            turn_gate: async_lock::Mutex::new(()),
        }
    }

    pub(crate) fn current_mode(&self) -> AcpMode {
        *self.mode.lock().expect("session mode mutex poisoned")
    }

    pub(crate) fn set_current_mode(&self, mode: AcpMode) {
        *self.mode.lock().expect("session mode mutex poisoned") = mode;
    }

    pub(crate) fn mode_state(&self) -> SessionModeState {
        self.current_mode().into_mode_state()
    }

    pub(crate) fn config_options(&self) -> Vec<ConfigOption> {
        let mode = self.current_mode();
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        build_config_options(mode, &meta.model, &meta.available_models, &meta.reasoning)
    }

    pub(crate) fn available_commands_update(&self) -> AvailableCommandsUpdate {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        AvailableCommandsUpdate {
            available_commands: build_available_commands(&meta.cli_commands),
            meta: None,
        }
    }

    pub(crate) fn supports_cli_command(&self, name: &str) -> bool {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        meta.cli_commands
            .iter()
            .any(|cmd| cmd.name.trim_start_matches('/') == name)
    }

    pub(crate) fn session_info(&self) -> SessionInfo {
        let meta = self.meta.lock().expect("session metadata mutex poisoned");
        SessionInfo {
            session_id: self.session_id.clone(),
            cwd: self.cwd.clone(),
            additional_directories: self.additional_directories.clone(),
            title: meta.title.clone(),
            updated_at: Some(meta.updated_at.clone()),
            meta: None,
            extra: BTreeMap::new(),
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

    /// Shut down the SDK client and drive its transport/process future to exit.
    pub(crate) async fn shutdown(&self) {
        self.turn_cancelled.store(true, Ordering::SeqCst);
        self.client.close();
        self.driver.drive_to_completion().await;
    }

    /// Execute one prompt turn on this session, streaming `SessionUpdate`s
    /// through `peer` until the CLI yields its terminal `ResultMessage`.
    pub(crate) async fn run_turn(
        &self,
        peer: &ClientPeer,
        auth_state: &Mutex<AuthState>,
        blocks: Vec<SdkContentBlock>,
        fallback_title_text: Option<&str>,
    ) -> Result<PromptResult> {
        let _turn_guard = self.turn_gate.lock().await;
        self.turn_cancelled.store(false, Ordering::SeqCst);

        let initial_title_update = {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            meta.updated_at = now_rfc3339();
            if meta.title.is_none()
                && let Some(raw_title) = fallback_title_text
                && let Some(clean_title) = derive_prompt_title(raw_title)
            {
                meta.title = Some(clean_title.clone());
                Some(SessionInfoUpdate {
                    title: Some(clean_title),
                    updated_at: Some(meta.updated_at.clone()),
                    meta: None,
                })
            } else {
                None
            }
        };

        if let Some(info_upd) = initial_title_update {
            send_update(
                peer,
                &self.session_id,
                SessionUpdate::SessionInfoUpdate(info_upd),
            )
            .await?;
        }

        Box::pin(self.driver.drive_until(async {
            self.client.send_user_blocks(blocks, None).await?;
            let mut streamer = TurnStreamer::default();

            loop {
                let next_ev = Box::pin(self.client.next_event()).await;
                match next_ev {
                    Ok(Some(event)) => {
                        if let Some(result) = Box::pin(self.handle_session_event(
                            peer,
                            auth_state,
                            &mut streamer,
                            Box::new(event),
                        ))
                        .await?
                        {
                            return Ok(result);
                        }
                    }
                    Ok(None) => {
                        if self.turn_cancelled.load(Ordering::SeqCst) {
                            return Ok(PromptResult {
                                stop_reason: StopReason::Cancelled,
                                meta: None,
                            });
                        }
                        return Err(ClaudeAcpError::Sdk(SdkError::ChannelClosed));
                    }
                    Err(err) => {
                        if self.turn_cancelled.load(Ordering::SeqCst) {
                            return Ok(PromptResult {
                                stop_reason: StopReason::Cancelled,
                                meta: None,
                            });
                        }
                        return Err(ClaudeAcpError::Sdk(err));
                    }
                }
            }
        }))
        .await
    }

    async fn handle_session_event(
        &self,
        peer: &ClientPeer,
        auth_state: &Mutex<AuthState>,
        streamer: &mut TurnStreamer,
        event: Box<SessionEvent>,
    ) -> Result<Option<PromptResult>> {
        match *event {
            SessionEvent::StreamEvent(msg) => {
                if msg.parent_tool_use_id.is_none() {
                    self.handle_stream_delta(peer, streamer, StreamDelta::from_stream_event(&msg))
                        .await?;
                }
            }
            SessionEvent::Assistant(assistant) => {
                if assistant.parent_tool_use_id.is_none() && assistant.agent_id.is_none() {
                    self.handle_assistant_message(peer, streamer, &assistant)
                        .await?;
                }
            }
            SessionEvent::ToolProgress(prog) => {
                self.handle_tool_progress(peer, streamer, &prog).await?;
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
                            send_update(peer, &self.session_id, SessionUpdate::ToolCallUpdate(upd))
                                .await?;
                        }
                    }
                }
            }
            SessionEvent::AuthStatus(auth_msg) => {
                let updated = auth_state
                    .lock()
                    .expect("auth state mutex poisoned")
                    .observe_cli_auth_status(&auth_msg);
                notify_auth_status(peer, &updated).await?;
            }
            SessionEvent::System(sys) => {
                self.handle_system_message(peer, streamer, &sys).await?;
            }
            SessionEvent::ConversationReset(_) => {
                self.permission_memory.clear();
                streamer.notice = TurnNotice::Reset;
            }
            SessionEvent::Other(other) => {
                self.handle_other_output(peer, &other).await?;
            }
            SessionEvent::Result(res) => {
                let final_res = self.finish_turn(peer, streamer, &res).await?;
                return Ok(Some(final_res));
            }
            SessionEvent::ApiError(_)
            | SessionEvent::RateLimit(_)
            | SessionEvent::CommandLifecycle(_)
            | SessionEvent::ToolUseSummary(_)
            | SessionEvent::PromptSuggestion(_) => {}
        }
        Ok(None)
    }

    async fn handle_stream_delta(
        &self,
        peer: &ClientPeer,
        streamer: &mut TurnStreamer,
        delta: Option<StreamDelta>,
    ) -> Result<()> {
        let Some(delta) = delta else {
            return Ok(());
        };
        match delta {
            StreamDelta::MessageStart { message_id } => {
                streamer.current_message_id = message_id;
                streamer.last_delta_stop_reason = None;
            }
            StreamDelta::Thinking { thinking, .. } => {
                if !thinking.is_empty() {
                    streamer.streamed_thinking_bytes += thinking.len();
                    send_update(
                        peer,
                        &self.session_id,
                        SessionUpdate::AgentThoughtChunk(acp_text_chunk(
                            thinking,
                            streamer.current_message_id.clone(),
                        )),
                    )
                    .await?;
                }
            }
            StreamDelta::Text { text, .. } => {
                if !text.is_empty() {
                    streamer.streamed_text_bytes += text.len();
                    streamer.text_state = TextProgress::Emitted;
                    send_update(
                        peer,
                        &self.session_id,
                        SessionUpdate::AgentMessageChunk(acp_text_chunk(
                            text,
                            streamer.current_message_id.clone(),
                        )),
                    )
                    .await?;
                }
            }
            StreamDelta::ToolUseStart { index, id, name } => {
                let is_todo = name == "TodoWrite";
                if !is_todo && streamer.announced_tools.insert(id.clone()) {
                    let call =
                        build_tool_call(id.clone(), &name, &Value::Null, ToolCallStatus::Pending);
                    send_update(peer, &self.session_id, SessionUpdate::ToolCall(call)).await?;
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
                    self.flush_partial_tool(peer, streamer, partial).await?;
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
        peer: &ClientPeer,
        streamer: &mut TurnStreamer,
        partial: PartialStreamTool,
    ) -> Result<()> {
        if partial.json_buf.trim().is_empty() {
            return Ok(());
        }
        let Ok(parsed_input) = serde_json::from_str::<Value>(&partial.json_buf) else {
            return Ok(());
        };

        if let Some(plan) = extract_todo_plan(&partial.name, &parsed_input) {
            if streamer.todo_ids.insert(partial.id) {
                send_update(peer, &self.session_id, SessionUpdate::Plan(plan)).await?;
            }
        } else if streamer.enriched_tools.insert(partial.id.clone()) {
            let upd = build_tool_input_update(
                partial.id.clone(),
                &partial.name,
                &parsed_input,
                ToolCallStatus::InProgress,
            );
            if let Some(diffs) = &upd.content {
                streamer.tool_diffs.insert(partial.id, diffs.clone());
            }
            send_update(peer, &self.session_id, SessionUpdate::ToolCallUpdate(upd)).await?;
        }
        Ok(())
    }

    async fn handle_assistant_message(
        &self,
        peer: &ClientPeer,
        streamer: &mut TurnStreamer,
        assistant: &claude_acp_sdk::AssistantMessage,
    ) -> Result<()> {
        streamer
            .last_assistant_stop_reason
            .clone_from(&assistant.message.stop_reason);
        streamer.last_assistant_refused = assistant.refused_message_id.is_some();

        let msg_id = streamer
            .current_message_id
            .clone()
            .or_else(|| (!assistant.message.id.is_empty()).then(|| assistant.message.id.clone()))
            .or_else(|| assistant.uuid.clone());

        // 1. De-duplicate Thinking blocks against prior `thinking_delta` chunks.
        let mut full_thinking = String::new();
        for block in &assistant.message.content {
            if let SdkContentBlock::Thinking(tb) = block {
                full_thinking.push_str(&tb.thinking);
            }
        }
        if !full_thinking.is_empty() {
            if streamer.streamed_thinking_bytes == 0 {
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::AgentThoughtChunk(acp_text_chunk(full_thinking, msg_id.clone())),
                )
                .await?;
            } else if full_thinking.len() > streamer.streamed_thinking_bytes
                && full_thinking.is_char_boundary(streamer.streamed_thinking_bytes)
            {
                let suffix = full_thinking[streamer.streamed_thinking_bytes..].to_string();
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::AgentThoughtChunk(acp_text_chunk(suffix, msg_id.clone())),
                )
                .await?;
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
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::AgentMessageChunk(acp_text_chunk(full_text, msg_id)),
                )
                .await?;
            } else if full_text.len() > streamer.streamed_text_bytes
                && full_text.is_char_boundary(streamer.streamed_text_bytes)
            {
                let suffix = full_text[streamer.streamed_text_bytes..].to_string();
                streamer.text_state = TextProgress::Emitted;
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::AgentMessageChunk(acp_text_chunk(suffix, msg_id)),
                )
                .await?;
            }
        }

        // 3. Emit or enrich tool calls & TodoWrite plans.
        for block in &assistant.message.content {
            match block {
                SdkContentBlock::ToolUse(tu) => {
                    self.emit_or_enrich_tool(peer, streamer, &tu.id, &tu.name, &tu.input)
                        .await?;
                }
                SdkContentBlock::McpToolUse(mcp) => {
                    self.emit_or_enrich_tool(peer, streamer, &mcp.id, &mcp.name, &mcp.input)
                        .await?;
                }
                SdkContentBlock::ServerToolUse(srv) => {
                    self.emit_or_enrich_tool(peer, streamer, &srv.id, &srv.name, &srv.input)
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
        Ok(())
    }

    async fn emit_or_enrich_tool(
        &self,
        peer: &ClientPeer,
        streamer: &mut TurnStreamer,
        id: &str,
        name: &str,
        input: &Value,
    ) -> Result<()> {
        if let Some(plan) = extract_todo_plan(name, input) {
            if streamer.todo_ids.insert(id.to_string()) {
                send_update(peer, &self.session_id, SessionUpdate::Plan(plan)).await?;
            }
            return Ok(());
        }

        if streamer.announced_tools.insert(id.to_string()) {
            streamer.enriched_tools.insert(id.to_string());
            let call = build_tool_call(id.to_string(), name, input, ToolCallStatus::InProgress);
            if !call.content.is_empty() {
                streamer
                    .tool_diffs
                    .insert(id.to_string(), call.content.clone());
            }
            send_update(peer, &self.session_id, SessionUpdate::ToolCall(call)).await?;
        } else if streamer.enriched_tools.insert(id.to_string()) {
            let upd =
                build_tool_input_update(id.to_string(), name, input, ToolCallStatus::InProgress);
            if let Some(diffs) = &upd.content {
                streamer.tool_diffs.insert(id.to_string(), diffs.clone());
            }
            send_update(peer, &self.session_id, SessionUpdate::ToolCallUpdate(upd)).await?;
        }
        Ok(())
    }

    async fn handle_tool_progress(
        &self,
        peer: &ClientPeer,
        streamer: &TurnStreamer,
        prog: &ToolProgressMessage,
    ) -> Result<()> {
        if prog.heartbeat == Some(true)
            || streamer.todo_ids.contains(&prog.tool_use_id)
            || !streamer.announced_tools.contains(&prog.tool_use_id)
        {
            return Ok(());
        }
        let upd = aither_acp::ToolCallUpdate {
            tool_call_id: prog.tool_use_id.clone(),
            status: Some(ToolCallStatus::InProgress),
            content: None,
            title: None,
            kind: None,
            locations: None,
            raw_input: None,
            raw_output: None,
            meta: None,
        };
        send_update(peer, &self.session_id, SessionUpdate::ToolCallUpdate(upd)).await
    }

    async fn handle_system_message(
        &self,
        peer: &ClientPeer,
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
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
                        current_mode_id: new_mode.as_str().to_string(),
                        meta: None,
                    }),
                )
                .await?;
            }
            if config_changed {
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate {
                        config_options: self.config_options(),
                        meta: None,
                    }),
                )
                .await?;
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
            self.apply_title_update(peer, new_title.to_string()).await?;
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

    async fn handle_other_output(&self, peer: &ClientPeer, output: &ClaudeOutput) -> Result<()> {
        let maybe_title = match output {
            ClaudeOutput::AiTitle(_) | ClaudeOutput::Summary(_) => {
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
            self.apply_title_update(peer, title).await?;
        }
        Ok(())
    }

    async fn apply_title_update(&self, peer: &ClientPeer, title: String) -> Result<()> {
        let updated_at = {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            meta.title = Some(title.clone());
            meta.updated_at = now_rfc3339();
            meta.updated_at.clone()
        };
        send_update(
            peer,
            &self.session_id,
            SessionUpdate::SessionInfoUpdate(SessionInfoUpdate {
                title: Some(title),
                updated_at: Some(updated_at),
                meta: None,
            }),
        )
        .await
    }

    async fn finish_turn(
        &self,
        peer: &ClientPeer,
        streamer: &mut TurnStreamer,
        res: &ResultMessage,
    ) -> Result<PromptResult> {
        let remaining: Vec<PartialStreamTool> = std::mem::take(&mut streamer.stream_tools)
            .into_values()
            .collect();
        for partial in remaining {
            self.flush_partial_tool(peer, streamer, partial).await?;
        }

        {
            let mut meta = self.meta.lock().expect("session metadata mutex poisoned");
            meta.updated_at = now_rfc3339();
        }

        if self.turn_cancelled.load(Ordering::SeqCst) {
            return Ok(PromptResult {
                stop_reason: StopReason::Cancelled,
                meta: None,
            });
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
                send_update(
                    peer,
                    &self.session_id,
                    SessionUpdate::AgentMessageChunk(acp_text_chunk(text, None)),
                )
                .await?;
            }
        }

        let stop_reason = map_stop_reason(
            res,
            streamer.last_assistant_stop_reason.as_ref(),
            streamer.last_delta_stop_reason.as_deref(),
            streamer.last_assistant_refused,
        );

        Ok(PromptResult {
            stop_reason,
            meta: None,
        })
    }
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

#[derive(Debug, Default)]
struct TurnStreamer {
    current_message_id: Option<String>,
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

pub async fn send_update(peer: &ClientPeer, session_id: &str, update: SessionUpdate) -> Result<()> {
    peer.session_update(SessionNotification::new(session_id, update))
        .await?;
    Ok(())
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
