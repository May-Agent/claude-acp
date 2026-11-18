//! [`ClaudeAgent`] implementing [`aither_acp::AgentHandler`].

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::future::Future;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use aither_acp::{
    AgentCapabilities, AgentHandler, AuthenticateParams, AuthenticateResult, ClientPeer,
    ConfigOptionUpdate, CurrentModeUpdate, InitializeParams, InitializeResult, JsonRpcError,
    LogoutParams, LogoutResult, McpCapabilities, PROTOCOL_VERSION, PromptCapabilities,
    PromptParams, PromptResult, SessionAdditionalDirectoriesCapabilities, SessionCancelParams,
    SessionCapabilities, SessionCloseCapabilities, SessionCloseParams, SessionCloseResult,
    SessionConfigValue, SessionDeleteCapabilities, SessionDeleteParams, SessionDeleteResult,
    SessionInfo, SessionListCapabilities, SessionListParams, SessionListResult, SessionLoadParams,
    SessionLoadResult, SessionNewParams, SessionNewResult, SessionResumeCapabilities,
    SessionResumeParams, SessionResumeResult, SessionSetConfigOptionParams,
    SessionSetConfigOptionResult, SessionSetModeParams, SessionSetModeResult, SessionUpdate,
    StopReason,
};
use chrono::{DateTime, Utc};
use claude_acp_sdk::{
    AnyControlHandler, ContentBlock as SdkContentBlock, SdkError, SpawnConfig, TextBlock,
    TranscriptStore, find_transcript_path,
};
use tracing::debug;
use uuid::Uuid;

use crate::auth::{AuthState, apply_auth_capabilities, build_auth_methods, notify_auth_status};
use crate::config::ClaudeAcpConfig;
use crate::convert::{
    acp_text_chunk, convert_mcp_servers, convert_prompt_blocks, transcript_to_updates,
};
use crate::driver::DriverSlot;
use crate::error::{ClaudeAcpError, Result};
use crate::mode::{AcpMode, ReasoningSetting};
use crate::permission::{PermissionMemory, SessionControlHandler};
use crate::session::{LiveSession, send_update};

struct AgentInner {
    config: ClaudeAcpConfig,
    auth: Mutex<AuthState>,
    sessions: Mutex<HashMap<String, Arc<LiveSession>>>,
}

/// ACP v1 agent handler bridging [`claude_acp_sdk`] onto [`aither_acp`].
///
/// Cheaply cloneable; pass directly to [`aither_acp::serve`] or construct and
/// run via [`crate::serve`] / [`crate::serve_stdio`].
#[derive(Clone)]
pub struct ClaudeAgent {
    inner: Arc<AgentInner>,
}

impl fmt::Debug for ClaudeAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let live_count = self
            .inner
            .sessions
            .lock()
            .map_or(0, |sessions| sessions.len());
        f.debug_struct("ClaudeAgent")
            .field("agent_info", &self.inner.config.agent_info)
            .field("live_sessions", &live_count)
            .finish_non_exhaustive()
    }
}

impl ClaudeAgent {
    /// Construct a new [`ClaudeAgent`] from `config`.
    #[must_use]
    pub fn new(config: ClaudeAcpConfig) -> Self {
        let auth = AuthState::from_spawn(&config.spawn);
        Self {
            inner: Arc::new(AgentInner {
                config,
                auth: Mutex::new(auth),
                sessions: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Shut down all active SDK sessions cleanly.
    pub async fn shutdown_all(&self) {
        let drained: Vec<Arc<LiveSession>> = {
            let mut guard = self
                .inner
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            guard.drain().map(|(_, session)| session).collect()
        };
        for session in drained {
            session.shutdown().await;
        }
    }

    fn resolve_transcript_store(&self) -> Result<TranscriptStore> {
        if let Some(home) = &self.inner.config.transcript_home {
            return Ok(TranscriptStore::new(home));
        }
        TranscriptStore::from_env().map_err(ClaudeAcpError::from)
    }

    fn resolve_initial_mode(&self) -> Result<AcpMode> {
        if let Some(mode_id) = &self.inner.config.default_mode {
            return AcpMode::parse(mode_id);
        }
        if let Some(perm_mode) = &self.inner.config.spawn.permission_mode {
            return Ok(AcpMode::from_permission_mode(perm_mode));
        }
        Ok(AcpMode::default())
    }

    fn get_live_session(&self, session_id: &str) -> Option<Arc<LiveSession>> {
        self.inner
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .get(session_id)
            .cloned()
    }

    fn remove_live_session(&self, session_id: &str) -> Option<Arc<LiveSession>> {
        self.inner
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .remove(session_id)
    }

    #[allow(clippy::too_many_arguments)]
    async fn start_sdk_session(
        &self,
        peer: &ClientPeer,
        session_id: String,
        pin_session_uuid: Option<Uuid>,
        resume_id: Option<String>,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_servers: &[aither_acp::McpServer],
        initial_title: Option<String>,
    ) -> Result<Arc<LiveSession>> {
        let initial_mode = self.resolve_initial_mode()?;
        let mode_shared = Arc::new(Mutex::new(initial_mode));
        let permission_memory = Arc::new(PermissionMemory::new());
        let turn_cancelled = Arc::new(AtomicBool::new(false));

        let mut spawn_cfg: SpawnConfig = self.inner.config.spawn.clone();
        spawn_cfg.cwd = Some(cwd.clone());
        spawn_cfg
            .additional_directories
            .extend(additional_directories.iter().cloned());
        spawn_cfg.include_partial_messages = self.inner.config.include_partial_messages;
        spawn_cfg.permission_mode = Some(initial_mode.to_permission_mode());

        if let Some(resume) = resume_id {
            spawn_cfg.resume = Some(resume);
            spawn_cfg.session_id = None;
        } else if let Some(uuid) = pin_session_uuid {
            spawn_cfg.session_id = Some(uuid);
            spawn_cfg.resume = None;
        }

        let converted_mcp = convert_mcp_servers(mcp_servers);
        let mcp_names: Vec<String> = converted_mcp.keys().cloned().collect();
        spawn_cfg.mcp_servers.extend(converted_mcp);

        {
            let auth = self.inner.auth.lock().expect("auth mutex poisoned");
            auth.apply_to_spawn(&mut spawn_cfg);
        }

        let initial_model = spawn_cfg
            .model
            .clone()
            .unwrap_or_else(|| "sonnet".to_string());
        let initial_reasoning =
            ReasoningSetting::from_spawn(spawn_cfg.effort.as_ref(), spawn_cfg.max_thinking_tokens);

        let control_handler = AnyControlHandler::new(SessionControlHandler::new(
            session_id.clone(),
            peer.clone(),
            Arc::clone(&permission_memory),
            Arc::clone(&mode_shared),
            Arc::clone(&turn_cancelled),
        ));

        let connected = self
            .inner
            .config
            .connector
            .connect(spawn_cfg, control_handler)
            .await?;

        let driver = DriverSlot::new(connected.driver);
        let client = connected.client;

        let init_response = driver.drive_until(client.initialize()).await?;

        // Replace any previously open session under the same ID cleanly.
        if let Some(prev) = self.remove_live_session(&session_id) {
            prev.shutdown().await;
        }

        let live = Arc::new(LiveSession::new(
            session_id.clone(),
            cwd,
            additional_directories,
            mcp_names,
            client,
            driver,
            mode_shared,
            initial_model,
            initial_reasoning,
            init_response,
            permission_memory,
            turn_cancelled,
            initial_title,
        ));

        self.inner
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .insert(session_id, Arc::clone(&live));

        Ok(live)
    }

    fn handle_initialize(&self, params: Option<&InitializeParams>) -> InitializeResult {
        let client_terminal_auth = params
            .and_then(|p| p.client_capabilities.auth.as_ref())
            .is_some_and(|auth| auth.terminal);

        let mut agent_capabilities = AgentCapabilities {
            load_session: true,
            prompt_capabilities: PromptCapabilities {
                image: true,
                audio: false,
                embedded_context: true,
            },
            mcp_capabilities: McpCapabilities {
                http: true,
                sse: true,
            },
            session_capabilities: SessionCapabilities {
                list: Some(SessionListCapabilities::default()),
                delete: Some(SessionDeleteCapabilities::default()),
                additional_directories: Some(SessionAdditionalDirectoriesCapabilities::default()),
                resume: Some(SessionResumeCapabilities::default()),
                close: Some(SessionCloseCapabilities::default()),
                ..SessionCapabilities::default()
            },
            ..AgentCapabilities::default()
        };
        apply_auth_capabilities(&mut agent_capabilities);

        let executable = self.inner.config.spawn.resolve_executable();
        let auth_methods = build_auth_methods(client_terminal_auth, &executable);

        InitializeResult {
            protocol_version: PROTOCOL_VERSION,
            agent_capabilities,
            agent_info: Some(self.inner.config.agent_info.clone()),
            auth_methods,
            meta: None,
        }
    }

    async fn handle_authenticate(
        &self,
        peer: &ClientPeer,
        params: AuthenticateParams,
    ) -> Result<AuthenticateResult> {
        let status = {
            let mut auth = self.inner.auth.lock().expect("auth mutex poisoned");
            auth.authenticate(&params)?
        };
        notify_auth_status(peer, &status).await?;
        Ok(AuthenticateResult::default())
    }

    async fn handle_logout(&self, peer: &ClientPeer) -> Result<LogoutResult> {
        let status = {
            let mut auth = self.inner.auth.lock().expect("auth mutex poisoned");
            auth.logout()
        };
        notify_auth_status(peer, &status).await?;
        Ok(LogoutResult::default())
    }

    async fn handle_new_session(
        &self,
        peer: &ClientPeer,
        params: SessionNewParams,
    ) -> Result<SessionNewResult> {
        validate_absolute_cwd(&params.cwd)?;
        let uuid = Uuid::new_v4();
        let session_id = uuid.to_string();

        let live = self
            .start_sdk_session(
                peer,
                session_id.clone(),
                Some(uuid),
                None,
                params.cwd,
                params.additional_directories,
                &params.mcp_servers,
                None,
            )
            .await?;

        send_update(
            peer,
            &session_id,
            SessionUpdate::AvailableCommandsUpdate(live.available_commands_update()),
        )
        .await?;

        Ok(SessionNewResult {
            session_id,
            modes: Some(live.mode_state()),
            config_options: Some(live.config_options()),
            meta: None,
            extra: BTreeMap::new(),
        })
    }

    async fn handle_load_session(
        &self,
        peer: &ClientPeer,
        params: SessionLoadParams,
    ) -> Result<SessionLoadResult> {
        validate_absolute_cwd(&params.cwd)?;
        let store = self.resolve_transcript_store()?;

        let entries = match store
            .load_transcript(&params.session_id, Some(&params.cwd))
            .await
        {
            Ok(entries) => entries,
            Err(SdkError::Io(io)) if io.kind() == ErrorKind::NotFound => {
                return Err(ClaudeAcpError::SessionNotFound(params.session_id));
            }
            Err(SdkError::Protocol(msg)) => {
                return Err(ClaudeAcpError::InvalidParams(msg));
            }
            Err(err) => return Err(ClaudeAcpError::Sdk(err)),
        };

        let persisted_title = store
            .list_sessions(Some(&params.cwd))
            .await
            .ok()
            .and_then(|list| {
                list.into_iter()
                    .find(|s| s.session_id == params.session_id)
                    .and_then(|s| s.title)
            });

        // Replay persisted transcript before attaching/returning `SessionLoadResult`.
        let replay_updates = transcript_to_updates(&entries);
        for update in replay_updates {
            send_update(peer, &params.session_id, update).await?;
        }

        let live = self
            .start_sdk_session(
                peer,
                params.session_id.clone(),
                None,
                Some(params.session_id.clone()),
                params.cwd,
                params.additional_directories,
                &params.mcp_servers,
                persisted_title,
            )
            .await?;

        send_update(
            peer,
            &params.session_id,
            SessionUpdate::AvailableCommandsUpdate(live.available_commands_update()),
        )
        .await?;

        Ok(SessionLoadResult {
            modes: Some(live.mode_state()),
            config_options: Some(live.config_options()),
            meta: None,
            extra: BTreeMap::new(),
        })
    }

    async fn handle_resume_session(
        &self,
        peer: &ClientPeer,
        params: SessionResumeParams,
    ) -> Result<SessionResumeResult> {
        validate_absolute_cwd(&params.cwd)?;
        let store = self.resolve_transcript_store()?;

        let found_on_disk =
            match find_transcript_path(store.home(), &params.session_id, Some(&params.cwd)).await {
                Ok(opt) => opt.is_some(),
                Err(SdkError::Protocol(msg)) => return Err(ClaudeAcpError::InvalidParams(msg)),
                Err(err) => return Err(ClaudeAcpError::Sdk(err)),
            };

        if !found_on_disk && self.get_live_session(&params.session_id).is_none() {
            return Err(ClaudeAcpError::SessionNotFound(params.session_id));
        }

        let persisted_title = store
            .list_sessions(Some(&params.cwd))
            .await
            .ok()
            .and_then(|list| {
                list.into_iter()
                    .find(|s| s.session_id == params.session_id)
                    .and_then(|s| s.title)
            });

        let live = self
            .start_sdk_session(
                peer,
                params.session_id.clone(),
                None,
                Some(params.session_id.clone()),
                params.cwd,
                params.additional_directories,
                &params.mcp_servers,
                persisted_title,
            )
            .await?;

        send_update(
            peer,
            &params.session_id,
            SessionUpdate::AvailableCommandsUpdate(live.available_commands_update()),
        )
        .await?;

        Ok(SessionResumeResult {
            modes: Some(live.mode_state()),
            config_options: Some(live.config_options()),
            meta: None,
            extra: BTreeMap::new(),
        })
    }

    async fn handle_list_sessions(&self, params: SessionListParams) -> Result<SessionListResult> {
        let offset = parse_cursor(params.cursor.as_deref())?;
        let store = self.resolve_transcript_store()?;
        let persisted = store.list_sessions(params.cwd.as_deref()).await?;

        let mut merged_map: HashMap<String, SessionInfo> = HashMap::new();

        for summary in persisted {
            let cwd = summary
                .cwd
                .or_else(|| params.cwd.clone())
                .unwrap_or_default();
            merged_map.insert(
                summary.session_id.clone(),
                SessionInfo {
                    session_id: summary.session_id,
                    cwd,
                    additional_directories: Vec::new(),
                    title: summary.title,
                    updated_at: summary.updated_at,
                    meta: None,
                    extra: BTreeMap::new(),
                },
            );
        }

        let live_infos: Vec<SessionInfo> = {
            let sessions = self.inner.sessions.lock().expect("sessions mutex poisoned");
            sessions
                .values()
                .filter(|s| {
                    params
                        .cwd
                        .as_deref()
                        .is_none_or(|filter_cwd| s.cwd == filter_cwd)
                })
                .map(|s| s.session_info())
                .collect()
        };

        for live_info in live_infos {
            merged_map
                .entry(live_info.session_id.clone())
                .and_modify(|existing| {
                    existing
                        .additional_directories
                        .clone_from(&live_info.additional_directories);
                    if live_info.title.is_some() {
                        existing.title.clone_from(&live_info.title);
                    }
                    if is_newer_timestamp(
                        live_info.updated_at.as_deref(),
                        existing.updated_at.as_deref(),
                    ) {
                        existing.updated_at.clone_from(&live_info.updated_at);
                    }
                })
                .or_insert(live_info);
        }

        let mut sorted: Vec<(Option<DateTime<Utc>>, SessionInfo)> = merged_map
            .into_values()
            .map(|info| {
                let ts = info
                    .updated_at
                    .as_deref()
                    .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));
                (ts, info)
            })
            .collect();

        sorted.sort_by(|(ts_a, info_a), (ts_b, info_b)| {
            Reverse(*ts_a)
                .cmp(&Reverse(*ts_b))
                .then_with(|| info_a.session_id.cmp(&info_b.session_id))
        });

        let all_sessions: Vec<SessionInfo> = sorted.into_iter().map(|(_, info)| info).collect();
        let page_size = self.inner.config.session_list_page_size.max(1);

        if offset >= all_sessions.len() {
            return Ok(SessionListResult {
                sessions: Vec::new(),
                next_cursor: None,
                meta: None,
            });
        }

        let end = (offset + page_size).min(all_sessions.len());
        let next_cursor = (end < all_sessions.len()).then(|| end.to_string());
        let page = all_sessions[offset..end].to_vec();

        Ok(SessionListResult {
            sessions: page,
            next_cursor,
            meta: None,
        })
    }

    async fn handle_close_session(&self, params: SessionCloseParams) -> Result<SessionCloseResult> {
        let Some(live) = self.remove_live_session(&params.session_id) else {
            return Err(ClaudeAcpError::SessionNotFound(params.session_id));
        };
        live.shutdown().await;
        Ok(SessionCloseResult::default())
    }

    async fn handle_delete_session(
        &self,
        params: SessionDeleteParams,
    ) -> Result<SessionDeleteResult> {
        let live_cwd = if let Some(live) = self.remove_live_session(&params.session_id) {
            let cwd = live.cwd.clone();
            live.shutdown().await;
            Some(cwd)
        } else {
            None
        };

        let store = self.resolve_transcript_store()?;
        let deleted_disk = match store
            .delete_transcript(&params.session_id, live_cwd.as_deref())
            .await
        {
            Ok(removed) => removed,
            Err(SdkError::Protocol(msg)) => return Err(ClaudeAcpError::InvalidParams(msg)),
            Err(err) => return Err(ClaudeAcpError::Sdk(err)),
        };

        if live_cwd.is_none() && !deleted_disk {
            return Err(ClaudeAcpError::SessionNotFound(params.session_id));
        }

        Ok(SessionDeleteResult::default())
    }

    async fn handle_prompt(&self, peer: &ClientPeer, params: PromptParams) -> Result<PromptResult> {
        let Some(session) = self.get_live_session(&params.session_id) else {
            return Err(ClaudeAcpError::SessionNotFound(params.session_id));
        };

        // Check for single-text slash commands (`/compact`, `/clear`, `/status`, `/logout`, `/review`).
        if let Some(slash) = parse_slash_command(&params.prompt) {
            return self.dispatch_slash_command(peer, &session, slash).await;
        }

        let sdk_blocks = convert_prompt_blocks(&params.prompt)?;
        let first_text = first_prompt_text(&params.prompt);
        session
            .run_turn(peer, &self.inner.auth, sdk_blocks, first_text)
            .await
    }

    async fn dispatch_slash_command(
        &self,
        peer: &ClientPeer,
        session: &Arc<LiveSession>,
        cmd: SlashCommand<'_>,
    ) -> Result<PromptResult> {
        match cmd.name {
            "status" => {
                let report = {
                    let auth = self.inner.auth.lock().expect("auth mutex poisoned");
                    session.format_status_report(&auth)
                };
                send_update(
                    peer,
                    &session.session_id,
                    SessionUpdate::AgentMessageChunk(acp_text_chunk(report, None)),
                )
                .await?;
                Ok(PromptResult {
                    stop_reason: StopReason::EndTurn,
                    meta: None,
                })
            }
            "logout" => {
                let status = {
                    let mut auth = self.inner.auth.lock().expect("auth mutex poisoned");
                    auth.logout()
                };
                notify_auth_status(peer, &status).await?;
                send_update(
                    peer,
                    &session.session_id,
                    SessionUpdate::AgentMessageChunk(acp_text_chunk(
                        "Signed out of Claude session credentials.".to_string(),
                        None,
                    )),
                )
                .await?;
                Ok(PromptResult {
                    stop_reason: StopReason::EndTurn,
                    meta: None,
                })
            }
            "clear" => {
                let blocks = vec![SdkContentBlock::Text(TextBlock {
                    text: "/clear".to_string(),
                    citations: Vec::new(),
                })];
                session.run_turn(peer, &self.inner.auth, blocks, None).await
            }
            "compact" => {
                let text = if cmd.args.is_empty() {
                    "/compact".to_string()
                } else {
                    format!("/compact {}", cmd.args)
                };
                let blocks = vec![SdkContentBlock::Text(TextBlock {
                    text,
                    citations: Vec::new(),
                })];
                session.run_turn(peer, &self.inner.auth, blocks, None).await
            }
            "review" => {
                let text = if session.supports_cli_command("review") {
                    if cmd.args.is_empty() {
                        "/review".to_string()
                    } else {
                        format!("/review {}", cmd.args)
                    }
                } else if cmd.args.is_empty() {
                    "Review the current workspace changes (git status and diff) for bugs, regressions, and missing test coverage."
                        .to_string()
                } else {
                    format!(
                        "Review `{}` for bugs, regressions, and missing test coverage.",
                        cmd.args
                    )
                };
                let blocks = vec![SdkContentBlock::Text(TextBlock {
                    text,
                    citations: Vec::new(),
                })];
                session.run_turn(peer, &self.inner.auth, blocks, None).await
            }
            _ => unreachable!("only intercepted commands are matched"),
        }
    }

    async fn handle_set_mode(
        &self,
        peer: &ClientPeer,
        params: SessionSetModeParams,
    ) -> Result<SessionSetModeResult> {
        let Some(session) = self.get_live_session(&params.session_id) else {
            return Err(ClaudeAcpError::SessionNotFound(params.session_id));
        };
        let target_mode = AcpMode::parse(&params.mode_id)?;

        session
            .driver
            .drive_until(
                session
                    .client
                    .set_permission_mode(target_mode.to_permission_mode()),
            )
            .await?;

        session.set_current_mode(target_mode);

        send_update(
            peer,
            &session.session_id,
            SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
                current_mode_id: target_mode.as_str().to_string(),
                meta: None,
            }),
        )
        .await?;

        send_update(
            peer,
            &session.session_id,
            SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate {
                config_options: session.config_options(),
                meta: None,
            }),
        )
        .await?;

        Ok(SessionSetModeResult::default())
    }

    async fn handle_set_config_option(
        &self,
        peer: &ClientPeer,
        params: SessionSetConfigOptionParams,
    ) -> Result<SessionSetConfigOptionResult> {
        let Some(session) = self.get_live_session(&params.session_id) else {
            return Err(ClaudeAcpError::SessionNotFound(params.session_id));
        };

        let SessionConfigValue::Select { value } = params.value else {
            return Err(ClaudeAcpError::InvalidParams(format!(
                "config option `{}` expects a string select value",
                params.config_id
            )));
        };

        match params.config_id.as_str() {
            "mode" => {
                let target_mode = AcpMode::parse(&value)?;
                session
                    .driver
                    .drive_until(
                        session
                            .client
                            .set_permission_mode(target_mode.to_permission_mode()),
                    )
                    .await?;
                session.set_current_mode(target_mode);
                send_update(
                    peer,
                    &session.session_id,
                    SessionUpdate::CurrentModeUpdate(CurrentModeUpdate {
                        current_mode_id: target_mode.as_str().to_string(),
                        meta: None,
                    }),
                )
                .await?;
            }
            "model" => {
                if value.trim().is_empty() {
                    return Err(ClaudeAcpError::InvalidParams(
                        "`model` value must not be empty".to_string(),
                    ));
                }
                let model_arg = (value != "default").then_some(value.as_str());
                session
                    .driver
                    .drive_until(session.client.set_model(model_arg))
                    .await?;
                session
                    .meta
                    .lock()
                    .expect("session metadata mutex poisoned")
                    .model
                    .clone_from(&value);
            }
            "reasoning_effort" => {
                apply_reasoning_option(&session, &value).await?;
            }
            other => {
                return Err(ClaudeAcpError::InvalidParams(format!(
                    "unknown config option `{other}`; expected `mode`, `model`, or `reasoning_effort`"
                )));
            }
        }

        let config_options = session.config_options();
        send_update(
            peer,
            &session.session_id,
            SessionUpdate::ConfigOptionUpdate(ConfigOptionUpdate {
                config_options: config_options.clone(),
                meta: None,
            }),
        )
        .await?;

        Ok(SessionSetConfigOptionResult {
            config_options,
            meta: None,
        })
    }
}

async fn apply_reasoning_option(session: &LiveSession, value: &str) -> Result<()> {
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

impl AgentHandler for ClaudeAgent {
    fn initialize(
        &self,
        _peer: &ClientPeer,
        params: Option<InitializeParams>,
    ) -> impl Future<Output = std::result::Result<InitializeResult, JsonRpcError>> + Send {
        std::future::ready(Ok(self.handle_initialize(params.as_ref())))
    }

    fn authenticate(
        &self,
        peer: &ClientPeer,
        params: AuthenticateParams,
    ) -> impl Future<Output = std::result::Result<AuthenticateResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move {
            this.handle_authenticate(&peer, params)
                .await
                .map_err(Into::into)
        }
    }

    fn logout(
        &self,
        peer: &ClientPeer,
        _params: LogoutParams,
    ) -> impl Future<Output = std::result::Result<LogoutResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move { this.handle_logout(&peer).await.map_err(Into::into) }
    }

    fn new_session(
        &self,
        peer: &ClientPeer,
        params: SessionNewParams,
    ) -> impl Future<Output = std::result::Result<SessionNewResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move {
            this.handle_new_session(&peer, params)
                .await
                .map_err(Into::into)
        }
    }

    fn load_session(
        &self,
        peer: &ClientPeer,
        params: SessionLoadParams,
    ) -> impl Future<Output = std::result::Result<SessionLoadResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move {
            this.handle_load_session(&peer, params)
                .await
                .map_err(Into::into)
        }
    }

    fn resume_session(
        &self,
        peer: &ClientPeer,
        params: SessionResumeParams,
    ) -> impl Future<Output = std::result::Result<SessionResumeResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move {
            this.handle_resume_session(&peer, params)
                .await
                .map_err(Into::into)
        }
    }

    fn list_sessions(
        &self,
        _peer: &ClientPeer,
        params: SessionListParams,
    ) -> impl Future<Output = std::result::Result<SessionListResult, JsonRpcError>> + Send {
        let this = self.clone();
        async move { this.handle_list_sessions(params).await.map_err(Into::into) }
    }

    fn close_session(
        &self,
        _peer: &ClientPeer,
        params: SessionCloseParams,
    ) -> impl Future<Output = std::result::Result<SessionCloseResult, JsonRpcError>> + Send {
        let this = self.clone();
        async move { this.handle_close_session(params).await.map_err(Into::into) }
    }

    fn delete_session(
        &self,
        _peer: &ClientPeer,
        params: SessionDeleteParams,
    ) -> impl Future<Output = std::result::Result<SessionDeleteResult, JsonRpcError>> + Send {
        let this = self.clone();
        async move { this.handle_delete_session(params).await.map_err(Into::into) }
    }

    fn prompt(
        &self,
        peer: &ClientPeer,
        params: PromptParams,
    ) -> impl Future<Output = std::result::Result<PromptResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move { this.handle_prompt(&peer, params).await.map_err(Into::into) }
    }

    fn set_mode(
        &self,
        peer: &ClientPeer,
        params: SessionSetModeParams,
    ) -> impl Future<Output = std::result::Result<SessionSetModeResult, JsonRpcError>> + Send {
        let this = self.clone();
        let peer = peer.clone();
        async move {
            this.handle_set_mode(&peer, params)
                .await
                .map_err(Into::into)
        }
    }

    fn set_config_option(
        &self,
        peer: &ClientPeer,
        params: SessionSetConfigOptionParams,
    ) -> impl Future<Output = std::result::Result<SessionSetConfigOptionResult, JsonRpcError>> + Send
    {
        let this = self.clone();
        let peer = peer.clone();
        async move {
            this.handle_set_config_option(&peer, params)
                .await
                .map_err(Into::into)
        }
    }

    fn cancel(
        &self,
        _peer: &ClientPeer,
        params: SessionCancelParams,
    ) -> impl Future<Output = ()> + Send {
        let this = self.clone();
        async move {
            let Some(session) = this.get_live_session(&params.session_id) else {
                debug!(session_id = %params.session_id, "ignoring session/cancel for inactive session");
                return;
            };
            session.turn_cancelled.store(true, Ordering::SeqCst);
            if let Err(err) = session.driver.drive_until(session.client.interrupt()).await {
                debug!(session_id = %params.session_id, %err, "interrupt after cancel ended with error");
            }
        }
    }
}

struct SlashCommand<'a> {
    name: &'a str,
    args: &'a str,
}

fn parse_slash_command(prompt: &[aither_acp::ContentBlock]) -> Option<SlashCommand<'_>> {
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
        "compact" | "clear" | "status" | "logout" | "review" => Some(SlashCommand { name, args }),
        _ => None,
    }
}

fn first_prompt_text(prompt: &[aither_acp::ContentBlock]) -> Option<&str> {
    prompt.iter().find_map(|block| match block {
        aither_acp::ContentBlock::Text(t) if !t.text.trim().is_empty() => Some(t.text.as_str()),
        _ => None,
    })
}

fn validate_absolute_cwd(cwd: &Path) -> Result<()> {
    if !cwd.is_absolute() {
        return Err(ClaudeAcpError::InvalidParams(format!(
            "`cwd` must be an absolute path, got `{}`",
            cwd.display()
        )));
    }
    Ok(())
}

fn parse_cursor(cursor: Option<&str>) -> Result<usize> {
    let Some(raw) = cursor else {
        return Ok(0);
    };
    raw.parse::<usize>()
        .map_err(|_| ClaudeAcpError::InvalidParams(format!("invalid pagination cursor `{raw}`")))
}

fn is_newer_timestamp(candidate: Option<&str>, existing: Option<&str>) -> bool {
    let Some(cand_str) = candidate else {
        return false;
    };
    let Some(exist_str) = existing else {
        return true;
    };
    match (
        DateTime::parse_from_rfc3339(cand_str),
        DateTime::parse_from_rfc3339(exist_str),
    ) {
        (Ok(cand_dt), Ok(exist_dt)) => cand_dt >= exist_dt,
        (Ok(_), Err(_)) => true,
        _ => false,
    }
}
