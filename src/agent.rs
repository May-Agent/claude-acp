//! [`ClaudeAgent`] implementing [`aither_acp::server::SessionHost`].

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use aither_acp::domain::{
    AgentCapabilities, EmptyObject, ExtMethod, McpCaps, PromptCaps, SessionCaps, SessionId,
    SessionInfo, SessionListParams, SessionPage, SessionSetupParams,
};
use aither_acp::rpc::{Peer, RpcError};
use aither_acp::server::{
    AuthConfig, AuthRequest, ResumeRequest, Server, ServerError, SessionContext, SessionHost,
};
use chrono::{DateTime, Utc};
use claude_acp_sdk::{
    AnyControlHandler, SdkError, SpawnConfig, TranscriptStore, TranscriptSync, find_transcript_path,
};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::auth::{AuthState, apply_auth_capabilities, build_auth_methods, notify_auth_status};
use crate::config::ClaudeAcpConfig;
use crate::convert::{convert_mcp_servers, transcript_to_updates};
use crate::driver::DriverSlot;
use crate::error::{ClaudeAcpError, Result};
use crate::mode::{AcpMode, ReasoningSetting};
use crate::permission::{PermissionMemory, SessionControlHandler};
use crate::session::{ClaudeSession, ClaudeSessionHandle};

/// ACP extension method querying MCP server status for a live session.
pub const EXT_MCP_STATUS: &str = "_claude/mcp_status";

/// ACP extension method querying context-window usage for a live session.
pub const EXT_CONTEXT_USAGE: &str = "_claude/context_usage";

/// ACP extension method querying the latest observed rate-limit snapshot for a live session.
pub const EXT_RATE_LIMITS: &str = "_claude/rate_limits";

/// ACP extension notification emitted whenever the Claude CLI reports a rate-limit event.
pub const NOTIFY_RATE_LIMIT: &str = "_claude/rate_limit";

pub struct AgentInner {
    pub(crate) config: ClaudeAcpConfig,
    pub(crate) auth: Mutex<AuthState>,
    pub(crate) sessions: Mutex<HashMap<String, ClaudeSessionHandle>>,
    /// Connection-level JSON-RPC peer (set by `serve` once the transport is up)
    /// used for `_auth/status_update` and other non-session notifications.
    pub(crate) peer: Mutex<Option<Peer>>,
}

impl AgentInner {
    pub(crate) fn resolve_transcript_store(&self) -> Result<TranscriptStore> {
        if let Some(home) = &self.config.transcript_home {
            return Ok(TranscriptStore::new(home));
        }
        TranscriptStore::from_env().map_err(ClaudeAcpError::from)
    }
}

impl fmt::Debug for AgentInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentInner").finish_non_exhaustive()
    }
}

/// ACP agent host bridging [`claude_acp_sdk`] onto [`aither_acp`].
///
/// Cheaply cloneable; pass directly to `Server::builder` or construct and run
/// via [`crate::serve`] / [`crate::serve_stdio`].
#[derive(Clone)]
pub struct ClaudeAgent {
    pub(crate) inner: Arc<AgentInner>,
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
                peer: Mutex::new(None),
            }),
        }
    }

    /// Attach the connection-level [`Peer`] (used for auth-status notifications).
    pub(crate) fn set_peer(&self, peer: Peer) {
        *self.inner.peer.lock().expect("peer mutex poisoned") = Some(peer);
    }

    /// The auth methods this agent advertises (`api-key`, `claude-login`).
    ///
    /// The `AuthConfig` API gives the host no view of client capabilities at
    /// `initialize` time, so `claude-login` is always exposed as an
    /// agent-managed method.
    #[must_use]
    pub fn auth_methods(&self) -> Vec<aither_acp::AuthMethod> {
        let executable = self.inner.config.spawn.resolve_executable();
        build_auth_methods(false, &executable)
    }

    /// Handle an [`AuthRequest`] (`auth/login`/`auth/logout`).
    ///
    /// # Errors
    ///
    /// Propagates authentication and notification failures as [`RpcError`].
    ///
    /// # Panics
    ///
    /// Panics if the internal auth-state mutex is poisoned.
    pub async fn handle_auth_request(&self, request: AuthRequest) -> Result<()> {
        let status = match request {
            AuthRequest::Login(req) => {
                let mut auth = self.inner.auth.lock().expect("auth mutex poisoned");
                auth.authenticate(req.method_id.as_str(), req.meta.as_ref())?
            }
            AuthRequest::Logout(_) => {
                let mut auth = self.inner.auth.lock().expect("auth mutex poisoned");
                auth.logout()
            }
        };
        let peer = self.inner.peer.lock().expect("peer mutex poisoned").clone();
        if let Some(peer) = peer {
            notify_auth_status(&peer, &status).await?;
        }
        Ok(())
    }

    /// Shut down all active SDK sessions cleanly.
    pub async fn shutdown_all(&self) {
        let drained: Vec<ClaudeSessionHandle> = {
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

    /// Build a configured [`aither_acp::Server`] wrapping this host with its
    /// auth and `_`-prefixed extension handlers (`_claude/mcp_status`,
    /// `_claude/context_usage`, `_claude/rate_limits`).
    #[must_use]
    pub fn into_server(self) -> Server {
        let agent_info = self.inner.config.agent_info.clone();
        let auth_agent = self.clone();
        let ext_agent = self.clone();

        Server::builder(self.clone(), agent_info)
            .auth(AuthConfig::new(self.auth_methods(), move |request| {
                let agent = auth_agent.clone();
                async move {
                    agent
                        .handle_auth_request(request)
                        .await
                        .map_err(ClaudeAcpError::into_rpc_error)
                }
            }))
            .ext(move |method, params| {
                let agent = ext_agent.clone();
                async move {
                    agent
                        .handle_ext(&method, params)
                        .await
                        .map_err(ClaudeAcpError::into_rpc_error)
                }
            })
            .build()
    }

    /// Handle `_`-prefixed extension methods (`_claude/mcp_status`,
    /// `_claude/context_usage`, `_claude/rate_limits`).
    ///
    /// # Errors
    ///
    /// Returns [`ClaudeAcpError::InvalidParams`] if `sessionId` is missing or
    /// empty, [`ClaudeAcpError::SessionNotFound`] if the session is not live,
    /// [`ClaudeAcpError::Sdk`] if the CLI control request fails, or a
    /// method-not-found RPC error for unrecognized extension methods.
    pub async fn handle_ext(&self, method: &ExtMethod, params: Option<Value>) -> Result<Value> {
        match method.as_str() {
            EXT_MCP_STATUS => {
                let session = self.require_live_session(params.as_ref())?;
                let status = session.mcp_status().await?;
                serde_json::to_value(status)
                    .map_err(|err| ClaudeAcpError::Sdk(SdkError::json("mcp_status", err)))
            }
            EXT_CONTEXT_USAGE => {
                let session = self.require_live_session(params.as_ref())?;
                let usage = session.context_usage().await?;
                serde_json::to_value(usage)
                    .map_err(|err| ClaudeAcpError::Sdk(SdkError::json("context_usage", err)))
            }
            EXT_RATE_LIMITS => {
                let session = self.require_live_session(params.as_ref())?;
                Ok(session.rate_limits_json())
            }
            _ => Err(ClaudeAcpError::Server(ServerError::Rpc(
                RpcError::method_not_found(method.as_str()),
            ))),
        }
    }

    fn require_live_session(&self, params: Option<&Value>) -> Result<ClaudeSessionHandle> {
        let params = params
            .ok_or_else(|| ClaudeAcpError::InvalidParams("missing extension params".to_string()))?;
        let session_id = params
            .get("sessionId")
            .or_else(|| params.get("session_id"))
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ClaudeAcpError::InvalidParams("missing `sessionId`".to_string()))?;
        self.get_live_session(session_id)
            .ok_or_else(|| ClaudeAcpError::SessionNotFound(session_id.to_string()))
    }

    fn resolve_transcript_store(&self) -> Result<TranscriptStore> {
        self.inner.resolve_transcript_store()
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

    fn get_live_session(&self, session_id: &str) -> Option<ClaudeSessionHandle> {
        self.inner
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .get(session_id)
            .cloned()
    }

    fn remove_live_session(&self, session_id: &str) -> Option<ClaudeSessionHandle> {
        self.inner
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .remove(session_id)
    }

    #[allow(clippy::too_many_arguments)]
    async fn start_sdk_session(
        &self,
        cx: &SessionContext,
        session_id: SessionId,
        pin_session_uuid: Option<Uuid>,
        resume_id: Option<String>,
        cwd: PathBuf,
        additional_directories: Vec<PathBuf>,
        mcp_servers: &[aither_acp::McpServer],
        initial_title: Option<String>,
    ) -> Result<ClaudeSessionHandle> {
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
        spawn_cfg.mcp_servers.extend(converted_mcp);
        let mcp_names: Vec<String> = spawn_cfg.mcp_servers.keys().cloned().collect();

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
            cx.clone(),
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
        if let Some(prev) = self.remove_live_session(session_id.as_str()) {
            prev.shutdown().await;
        }

        let live = ClaudeSessionHandle(Arc::new(ClaudeSession::new(
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
            Arc::downgrade(&self.inner),
        )));

        self.inner
            .sessions
            .lock()
            .expect("sessions mutex poisoned")
            .insert(session_id.as_str().to_string(), live.clone());

        Ok(live)
    }

    async fn load_persisted_title(&self, session_id: &str, cwd: &Path) -> Option<String> {
        let store = self.resolve_transcript_store().ok()?;
        store.list_sessions(Some(cwd)).await.ok().and_then(|list| {
            list.into_iter()
                .find(|s| s.session_id == session_id)
                .and_then(|s| s.title)
        })
    }
}

impl SessionHost for ClaudeAgent {
    type Session = ClaudeSessionHandle;

    fn capabilities(&self) -> AgentCapabilities {
        let mut caps = AgentCapabilities {
            session: Some(SessionCaps {
                prompt: Some(PromptCaps {
                    image: Some(EmptyObject),
                    embedded_context: Some(EmptyObject),
                    ..PromptCaps::default()
                }),
                mcp: Some(McpCaps {
                    stdio: Some(EmptyObject),
                    http: Some(EmptyObject),
                    // v1 `sse` MCP servers are still honored inbound; they
                    // cannot be advertised through the domain capability set.
                    extra: {
                        let mut extra = BTreeMap::new();
                        extra.insert("sse".to_string(), json!({}));
                        extra
                    },
                    ..McpCaps::default()
                }),
                delete: Some(EmptyObject),
                additional_directories: Some(EmptyObject),
                options: Some(EmptyObject),
                title: Some(EmptyObject),
                extra: {
                    let mut extra = BTreeMap::new();
                    extra.insert("list".to_string(), json!({}));
                    extra.insert("resume".to_string(), json!({}));
                    extra.insert("close".to_string(), json!({}));
                    extra
                },
                ..SessionCaps::default()
            }),
            ..AgentCapabilities::default()
        };
        apply_auth_capabilities(&mut caps);
        caps
    }

    async fn new_session(
        &self,
        params: SessionSetupParams,
        cx: &SessionContext,
    ) -> std::result::Result<Self::Session, ServerError> {
        let run = async {
            validate_absolute_cwd(&params.cwd)?;
            let session_id = cx.session_id().clone();
            let pin_uuid = Uuid::parse_str(session_id.as_str()).ok();
            self.start_sdk_session(
                cx,
                session_id,
                pin_uuid,
                None,
                params.cwd,
                params.additional_directories,
                &params.mcp_servers,
                None,
            )
            .await
        };
        run.await.map_err(|e| ServerError::Rpc(e.into_rpc_error()))
    }

    async fn resume_session(
        &self,
        req: ResumeRequest,
        cx: &SessionContext,
    ) -> std::result::Result<Self::Session, ServerError> {
        let run = async {
            let params = req.params;
            let session_id = params.session_id.clone();
            validate_absolute_cwd(&params.setup.cwd)?;
            validate_session_id(session_id.as_str())?;
            let store = self.resolve_transcript_store()?;

            if let Err(err) = self
                .inner
                .config
                .transcript_sync
                .pull(session_id.as_str(), &params.setup.cwd, store.home())
                .await
            {
                return match err {
                    SdkError::Protocol(msg) => Err(ClaudeAcpError::InvalidParams(msg)),
                    other => Err(ClaudeAcpError::Sdk(other)),
                };
            }

            let want_replay = params.replay_from.is_some();

            if want_replay {
                // `session/load` semantics: replay the persisted transcript.
                let entries = match store
                    .load_transcript(session_id.as_str(), Some(&params.setup.cwd))
                    .await
                {
                    Ok(entries) => entries,
                    Err(SdkError::Io(io)) if io.kind() == ErrorKind::NotFound => {
                        return Err(ClaudeAcpError::SessionNotFound(
                            session_id.as_str().to_string(),
                        ));
                    }
                    Err(SdkError::Protocol(msg)) => {
                        return Err(ClaudeAcpError::InvalidParams(msg));
                    }
                    Err(err) => return Err(ClaudeAcpError::Sdk(err)),
                };

                for update in transcript_to_updates(&entries) {
                    let _ = req.replay.send(update);
                }
            } else {
                // `session/resume` semantics: verify the session exists.
                let found_on_disk = match find_transcript_path(
                    store.home(),
                    session_id.as_str(),
                    Some(&params.setup.cwd),
                )
                .await
                {
                    Ok(opt) => opt.is_some(),
                    Err(SdkError::Protocol(msg)) => {
                        return Err(ClaudeAcpError::InvalidParams(msg));
                    }
                    Err(err) => return Err(ClaudeAcpError::Sdk(err)),
                };

                if !found_on_disk && self.get_live_session(session_id.as_str()).is_none() {
                    return Err(ClaudeAcpError::SessionNotFound(
                        session_id.as_str().to_string(),
                    ));
                }
            }

            let persisted_title = self
                .load_persisted_title(session_id.as_str(), &params.setup.cwd)
                .await;

            self.start_sdk_session(
                cx,
                session_id.clone(),
                None,
                Some(session_id.as_str().to_string()),
                params.setup.cwd,
                params.setup.additional_directories,
                &params.setup.mcp_servers,
                persisted_title,
            )
            .await
        };
        run.await.map_err(|e| ServerError::Rpc(e.into_rpc_error()))
    }

    async fn list_sessions(
        &self,
        params: SessionListParams,
    ) -> std::result::Result<SessionPage, RpcError> {
        let run = async {
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
                        session_id: SessionId::from_string(summary.session_id),
                        cwd,
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
                    .entry(live_info.session_id.as_str().to_string())
                    .and_modify(|existing| {
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
                    .then_with(|| info_a.session_id.as_str().cmp(info_b.session_id.as_str()))
            });

            let all_sessions: Vec<SessionInfo> = sorted.into_iter().map(|(_, info)| info).collect();
            let page_size = self.inner.config.session_list_page_size.max(1);

            if offset >= all_sessions.len() {
                return Ok(SessionPage::default());
            }

            let end = (offset + page_size).min(all_sessions.len());
            let next_cursor = (end < all_sessions.len()).then(|| end.to_string());
            let page = all_sessions[offset..end].to_vec();

            Ok(SessionPage {
                sessions: page,
                next_cursor,
                ..SessionPage::default()
            })
        };
        run.await.map_err(ClaudeAcpError::into_rpc_error)
    }

    async fn delete_session(&self, session_id: &SessionId) -> std::result::Result<(), RpcError> {
        let run = async {
            let live_cwd = if let Some(live) = self.remove_live_session(session_id.as_str()) {
                let cwd = live.cwd.clone();
                live.shutdown().await;
                Some(cwd)
            } else {
                None
            };

            let store = self.resolve_transcript_store()?;
            let deleted_disk = match store
                .delete_transcript(session_id.as_str(), live_cwd.as_deref())
                .await
            {
                Ok(removed) => removed,
                Err(SdkError::Protocol(msg)) => return Err(ClaudeAcpError::InvalidParams(msg)),
                Err(err) => return Err(ClaudeAcpError::Sdk(err)),
            };

            if live_cwd.is_none() && !deleted_disk {
                return Err(ClaudeAcpError::SessionNotFound(
                    session_id.as_str().to_string(),
                ));
            }

            Ok(())
        };
        run.await.map_err(ClaudeAcpError::into_rpc_error)
    }

    async fn shutdown(&self) {
        self.shutdown_all().await;
    }
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

fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty()
        || session_id == "."
        || session_id == ".."
        || session_id.contains('/')
        || session_id.contains('\\')
        || session_id.contains('\0')
    {
        return Err(ClaudeAcpError::InvalidParams(format!(
            "invalid session_id `{session_id}`"
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
