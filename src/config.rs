//! Server configuration and pluggable SDK session connector traits.

use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use aither_acp::Implementation;
use claude_acp_sdk::{
    AnyControlHandler, AnyTranscriptSync, Client, ControlHandler, NoopTranscriptSync, SpawnConfig,
    TranscriptSync,
};
use futures_lite::io::{AsyncRead, AsyncWrite};
use tracing::debug;

/// Default maximum number of session entries returned per `session/list` page.
const DEFAULT_SESSION_LIST_PAGE_SIZE: usize = 50;

/// Connected `claude-acp-sdk` [`Client`] paired with its background I/O driver.
pub struct ConnectedSession {
    /// Cloneable SDK client handle for control requests, prompts, and events.
    pub client: Client,
    pub(crate) driver: Pin<Box<dyn Future<Output = ()> + Send>>,
}

impl fmt::Debug for ConnectedSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConnectedSession")
            .field("client", &self.client)
            .finish_non_exhaustive()
    }
}

impl ConnectedSession {
    /// Pair an existing [`Client`] with its companion `driver` future.
    #[must_use]
    pub fn new<F, T>(client: Client, driver: F) -> Self
    where
        F: Future<Output = claude_acp_sdk::Result<T>> + Send + 'static,
    {
        Self {
            client,
            driver: Box::pin(async move {
                if let Err(err) = driver.await {
                    debug!(%err, "Claude SDK session driver terminated with error");
                }
            }),
        }
    }

    /// Attach a [`Client`] directly to an `AsyncRead + AsyncWrite` stream pair
    /// (for example an in-memory mock Claude CLI pipe in integration tests).
    #[must_use]
    pub fn attach<R, W, H>(read: R, write: W, handler: H) -> Self
    where
        R: AsyncRead + Unpin + Send + 'static,
        W: AsyncWrite + Unpin + Send + 'static,
        H: ControlHandler,
    {
        let (client, driver) = Client::attach(read, write, handler);
        Self::new(client, driver)
    }
}

/// Factory that launches or attaches a [`ConnectedSession`] for each ACP
/// `session/new`, `session/load`, or `session/resume` request.
///
/// Production servers use [`ProcessSessionConnector`] (enabled by the default
/// `process` cargo feature), while deterministic integration tests inject an
/// in-memory mock-CLI connector via [`ClaudeAcpConfig::connector`].
pub trait SessionConnector: Send + Sync + 'static {
    /// Establish a Claude Code SDK session from `config` using `handler` for
    /// inbound CLI control requests (`can_use_tool`, hooks, SDK MCP).
    fn connect(
        &self,
        config: SpawnConfig,
        handler: AnyControlHandler,
    ) -> impl Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send;
}

impl<F, Fut> SessionConnector for F
where
    F: Fn(SpawnConfig, AnyControlHandler) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send + 'static,
{
    fn connect(
        &self,
        config: SpawnConfig,
        handler: AnyControlHandler,
    ) -> impl Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send {
        (self)(config, handler)
    }
}

/// Default [`SessionConnector`] that spawns a local `claude` CLI child process
/// via [`Client::spawn`].
#[cfg(feature = "process")]
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessSessionConnector;

#[cfg(feature = "process")]
impl SessionConnector for ProcessSessionConnector {
    fn connect(
        &self,
        config: SpawnConfig,
        handler: AnyControlHandler,
    ) -> impl Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send {
        let res = Client::spawn(config, handler)
            .map(|(client, driver)| ConnectedSession::new(client, driver));
        std::future::ready(res)
    }
}

#[cfg(not(feature = "process"))]
#[derive(Debug, Clone, Copy, Default)]
struct DisabledProcessConnector;

#[cfg(not(feature = "process"))]
impl SessionConnector for DisabledProcessConnector {
    fn connect(
        &self,
        _config: SpawnConfig,
        _handler: AnyControlHandler,
    ) -> impl Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send {
        std::future::ready(Err(claude_acp_sdk::SdkError::Protocol(
            "spawning a `claude` child process requires the `process` cargo feature; configure a custom `SessionConnector` via `ClaudeAcpConfig::connector`"
                .to_string(),
        )))
    }
}

trait SessionConnectorImpl: Send + Sync + 'static {
    fn connect_dyn<'a>(
        &'a self,
        config: SpawnConfig,
        handler: AnyControlHandler,
    ) -> Pin<Box<dyn Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send + 'a>>;
}

impl<C: SessionConnector> SessionConnectorImpl for C {
    fn connect_dyn<'a>(
        &'a self,
        config: SpawnConfig,
        handler: AnyControlHandler,
    ) -> Pin<Box<dyn Future<Output = claude_acp_sdk::Result<ConnectedSession>> + Send + 'a>> {
        Box::pin(self.connect(config, handler))
    }
}

#[derive(Clone)]
pub struct AnySessionConnector {
    inner: Arc<dyn SessionConnectorImpl>,
}

impl fmt::Debug for AnySessionConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnySessionConnector")
            .finish_non_exhaustive()
    }
}

impl AnySessionConnector {
    pub(crate) fn new<C: SessionConnector>(connector: C) -> Self {
        Self {
            inner: Arc::new(connector),
        }
    }

    pub(crate) async fn connect(
        &self,
        config: SpawnConfig,
        handler: AnyControlHandler,
    ) -> claude_acp_sdk::Result<ConnectedSession> {
        self.inner.connect_dyn(config, handler).await
    }
}

fn default_connector() -> AnySessionConnector {
    #[cfg(feature = "process")]
    {
        AnySessionConnector::new(ProcessSessionConnector)
    }
    #[cfg(not(feature = "process"))]
    {
        AnySessionConnector::new(DisabledProcessConnector)
    }
}

/// Configuration for [`ClaudeAgent`](crate::ClaudeAgent), [`serve`](crate::serve),
/// and [`serve_stdio`](crate::serve_stdio).
#[derive(Debug, Clone)]
pub struct ClaudeAcpConfig {
    /// Base [`SpawnConfig`] template applied to newly created or resumed sessions.
    pub spawn: SpawnConfig,
    /// Optional initial ACP mode identifier (`"read-only"`, `"workspace-write"`,
    /// `"agent"`, `"auto"`, `"dont-ask"`, or `"agent-full-access"`). Defaults
    /// to `"agent"` unless `spawn.permission_mode` specifies another mode.
    pub default_mode: Option<String>,
    /// Whether to stream incremental token/thinking deltas (`--include-partial-messages`).
    pub include_partial_messages: bool,
    /// Optional home directory override for [`TranscriptStore`](claude_acp_sdk::TranscriptStore)
    /// (`<home>/.claude/projects`), primarily used in isolated tests.
    pub transcript_home: Option<PathBuf>,
    /// Pluggable lifecycle hook synchronizing session transcripts (main and
    /// `subagents/agent-*.jsonl` files) before resume/load and after turns and
    /// shutdown. Defaults to [`NoopTranscriptSync`].
    pub transcript_sync: AnyTranscriptSync,
    /// Maximum number of sessions returned per `session/list` page.
    pub session_list_page_size: usize,
    /// Implementation metadata advertised in `initialize`.
    pub agent_info: Implementation,
    pub(crate) connector: AnySessionConnector,
}

impl Default for ClaudeAcpConfig {
    fn default() -> Self {
        Self {
            spawn: SpawnConfig::new().include_partial_messages(true),
            default_mode: None,
            include_partial_messages: true,
            transcript_home: None,
            transcript_sync: AnyTranscriptSync::new(NoopTranscriptSync),
            session_list_page_size: DEFAULT_SESSION_LIST_PAGE_SIZE,
            agent_info: Implementation {
                name: env!("CARGO_PKG_NAME").to_string(),
                title: Some("Claude Code".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                meta: None,
                extra: std::collections::BTreeMap::new(),
            },
            connector: default_connector(),
        }
    }
}

impl ClaudeAcpConfig {
    /// Construct a default [`ClaudeAcpConfig`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the base [`SpawnConfig`] template.
    #[must_use]
    pub fn spawn(mut self, spawn: SpawnConfig) -> Self {
        self.include_partial_messages = spawn.include_partial_messages;
        self.spawn = spawn;
        self
    }

    /// Override the `claude` CLI executable path on [`Self::spawn`].
    #[must_use]
    pub fn claude_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.spawn.claude_path = Some(path.into());
        self
    }

    /// Set the default ACP session mode identifier (`"read-only"`,
    /// `"workspace-write"`, `"agent"`, `"auto"`, `"dont-ask"`, or
    /// `"agent-full-access"`).
    #[must_use]
    pub fn default_mode(mut self, mode: impl Into<String>) -> Self {
        self.default_mode = Some(mode.into());
        self
    }

    /// Enable or disable incremental streaming deltas (`--include-partial-messages`).
    #[must_use]
    pub const fn include_partial_messages(mut self, enabled: bool) -> Self {
        self.include_partial_messages = enabled;
        self.spawn.include_partial_messages = enabled;
        self
    }

    /// Override the base home directory used by [`TranscriptStore`](claude_acp_sdk::TranscriptStore).
    #[must_use]
    pub fn transcript_home(mut self, home: impl Into<PathBuf>) -> Self {
        self.transcript_home = Some(home.into());
        self
    }

    /// Install a custom [`TranscriptSync`] lifecycle hook for remote transcript
    /// pull/push synchronization.
    #[must_use]
    pub fn transcript_sync<S: TranscriptSync>(mut self, sync: S) -> Self {
        self.transcript_sync = AnyTranscriptSync::new(sync);
        self
    }

    /// Set the page size used by `session/list` cursor pagination.
    #[must_use]
    pub const fn session_list_page_size(mut self, page_size: usize) -> Self {
        self.session_list_page_size = page_size;
        self
    }

    /// Override the [`Implementation`] metadata advertised in `initialize`.
    #[must_use]
    pub fn agent_info(mut self, info: Implementation) -> Self {
        self.agent_info = info;
        self
    }

    /// Inject a custom [`SessionConnector`] (e.g. for deterministic in-memory
    /// mock-CLI testing without spawning a `claude` process).
    #[must_use]
    pub fn connector<C: SessionConnector>(mut self, connector: C) -> Self {
        self.connector = AnySessionConnector::new(connector);
        self
    }
}
