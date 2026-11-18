//! Inbound CLI-to-SDK control-request handler trait and permission decision types.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use claude_codes::{
    ControlCancelRequest, HookCallbackRequest, McpMessageRequest, Permission, PermissionResult,
    ToolPermissionRequest,
};
use serde_json::{Map, Value};

use crate::error::{Result, SdkError};

/// Decision returned by [`ControlHandler::can_use_tool`] for a CLI tool
/// permission prompt.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionDecision {
    /// Approve tool execution, optionally rewriting the tool input or
    /// persisting updated session/project permission rules.
    Allow {
        /// Replacement tool input object; when `None`, the original
        /// [`ToolPermissionRequest::input`] is echoed back as required by the
        /// Claude CLI wire contract.
        updated_input: Option<Value>,
        /// Additional permission grants (`addRules` / `setMode`) to apply.
        updated_permissions: Vec<Permission>,
    },
    /// Reject tool execution with a human/model-readable explanation.
    Deny {
        /// Reason shown to the model explaining why the tool call was denied.
        message: String,
        /// When `true`, immediately interrupts the active turn as well.
        interrupt: bool,
    },
}

impl PermissionDecision {
    /// Approve the tool execution with its unmodified input.
    #[must_use]
    pub const fn allow() -> Self {
        Self::Allow {
            updated_input: None,
            updated_permissions: Vec::new(),
        }
    }

    /// Approve the tool execution with a modified input JSON object.
    #[must_use]
    pub const fn allow_with_input(updated_input: Value) -> Self {
        Self::Allow {
            updated_input: Some(updated_input),
            updated_permissions: Vec::new(),
        }
    }

    /// Attach replacement input to an [`Allow`](Self::Allow) decision.
    #[must_use]
    pub fn with_updated_input(mut self, input: Value) -> Self {
        if let Self::Allow { updated_input, .. } = &mut self {
            *updated_input = Some(input);
        }
        self
    }

    /// Attach remembered [`Permission`] rules to an [`Allow`](Self::Allow)
    /// decision.
    #[must_use]
    pub fn with_permissions<I>(mut self, permissions: I) -> Self
    where
        I: IntoIterator<Item = Permission>,
    {
        if let Self::Allow {
            updated_permissions,
            ..
        } = &mut self
        {
            updated_permissions.extend(permissions);
        }
        self
    }

    /// Reject the tool execution without interrupting the turn.
    #[must_use]
    pub fn deny(message: impl Into<String>) -> Self {
        Self::Deny {
            message: message.into(),
            interrupt: false,
        }
    }

    /// Reject the tool execution and interrupt the active turn (`interrupt: true`).
    #[must_use]
    pub fn deny_and_interrupt(message: impl Into<String>) -> Self {
        Self::Deny {
            message: message.into(),
            interrupt: true,
        }
    }

    /// Set the `interrupt` flag on a [`Deny`](Self::Deny) decision.
    #[must_use]
    pub const fn with_interrupt(mut self, should_interrupt: bool) -> Self {
        if let Self::Deny { interrupt, .. } = &mut self {
            *interrupt = should_interrupt;
        }
        self
    }

    /// Convert into the `claude-codes` wire [`PermissionResult`], filling in
    /// `original_input` when `updated_input` was left `None`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Json`] if serializing `updated_permissions` fails.
    pub fn into_permission_result(self, original_input: Value) -> Result<PermissionResult> {
        match self {
            Self::Allow {
                updated_input,
                updated_permissions,
            } => {
                let input = updated_input.unwrap_or(original_input);
                if updated_permissions.is_empty() {
                    Ok(PermissionResult::allow(input))
                } else {
                    let serialized = updated_permissions
                        .into_iter()
                        .map(|perm| {
                            serde_json::to_value(perm)
                                .map_err(|err| SdkError::json("updated_permissions", err))
                        })
                        .collect::<Result<Vec<Value>>>()?;
                    Ok(PermissionResult::allow_with_permissions(input, serialized))
                }
            }
            Self::Deny { message, interrupt } => Ok(PermissionResult::Deny { message, interrupt }),
        }
    }
}

/// Handles inbound `control_request` callbacks initiated by the Claude CLI.
///
/// Methods are polled concurrently on the session driver task so a long-lived
/// permission prompt never blocks outbound `interrupt` writes or unrelated
/// stream events.
pub trait ControlHandler: Send + Sync + 'static {
    /// Evaluate a `can_use_tool` permission prompt.
    ///
    /// The default denies execution with `interrupt: false` so an unconfigured
    /// handler never auto-approves privileged tools.
    fn can_use_tool(
        &self,
        req: ToolPermissionRequest,
    ) -> impl Future<Output = Result<PermissionDecision>> + Send {
        async move {
            Ok(PermissionDecision::deny(format!(
                "tool `{}` denied: no permission handler configured",
                req.tool_name
            )))
        }
    }

    /// Handle a `hook_callback` request and return the hook JSON payload.
    ///
    /// Defaults to an empty JSON object `{}` (continue without modification).
    fn hook_callback(
        &self,
        _req: HookCallbackRequest,
    ) -> impl Future<Output = Result<Value>> + Send {
        std::future::ready(Ok(Value::Object(Map::new())))
    }

    /// Route an `mcp_message` JSON-RPC frame to an SDK-hosted MCP server and
    /// return its JSON-RPC response object.
    ///
    /// The session driver wraps the returned value under `{"mcp_response": ...}`
    /// in the `control_response` frame.
    fn mcp_message(&self, req: McpMessageRequest) -> impl Future<Output = Result<Value>> + Send {
        async move {
            Err(SdkError::Protocol(format!(
                "no SDK MCP server registered for `{}`",
                req.server_name
            )))
        }
    }

    /// Observe a `control_cancel_request` withdrawing a previously issued CLI
    /// `control_request` (after the driver cancels the in-flight handler future).
    fn cancel_request(&self, _req: ControlCancelRequest) -> impl Future<Output = ()> + Send {
        std::future::ready(())
    }
}

/// Default [`ControlHandler`] that denies unconfigured tool prompts, returns
/// empty hook acknowledgments, and rejects unknown SDK MCP servers.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopControlHandler;

impl ControlHandler for NoopControlHandler {}

trait ControlHandlerImpl: Send + Sync + 'static {
    fn can_use_tool_dyn<'a>(
        &'a self,
        req: ToolPermissionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<PermissionDecision>> + Send + 'a>>;

    fn hook_callback_dyn<'a>(
        &'a self,
        req: HookCallbackRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;

    fn mcp_message_dyn<'a>(
        &'a self,
        req: McpMessageRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>>;

    fn cancel_request_dyn<'a>(
        &'a self,
        req: ControlCancelRequest,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
}

impl<H: ControlHandler> ControlHandlerImpl for H {
    fn can_use_tool_dyn<'a>(
        &'a self,
        req: ToolPermissionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<PermissionDecision>> + Send + 'a>> {
        Box::pin(self.can_use_tool(req))
    }

    fn hook_callback_dyn<'a>(
        &'a self,
        req: HookCallbackRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(self.hook_callback(req))
    }

    fn mcp_message_dyn<'a>(
        &'a self,
        req: McpMessageRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Value>> + Send + 'a>> {
        Box::pin(self.mcp_message(req))
    }

    fn cancel_request_dyn<'a>(
        &'a self,
        req: ControlCancelRequest,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> {
        Box::pin(self.cancel_request(req))
    }
}

/// Type-erased wrapper around any [`ControlHandler`] keeping `Pin<Box<dyn Future>>`
/// strictly internal.
#[derive(Clone)]
pub struct AnyControlHandler {
    inner: Arc<dyn ControlHandlerImpl>,
}

impl fmt::Debug for AnyControlHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnyControlHandler").finish_non_exhaustive()
    }
}

impl AnyControlHandler {
    /// Erase the concrete type of `handler` behind [`AnyControlHandler`].
    #[must_use]
    pub fn new<H: ControlHandler>(handler: H) -> Self {
        Self {
            inner: Arc::new(handler),
        }
    }
}

impl ControlHandler for AnyControlHandler {
    fn can_use_tool(
        &self,
        req: ToolPermissionRequest,
    ) -> impl Future<Output = Result<PermissionDecision>> + Send {
        let inner = Arc::clone(&self.inner);
        async move { inner.can_use_tool_dyn(req).await }
    }

    fn hook_callback(
        &self,
        req: HookCallbackRequest,
    ) -> impl Future<Output = Result<Value>> + Send {
        let inner = Arc::clone(&self.inner);
        async move { inner.hook_callback_dyn(req).await }
    }

    fn mcp_message(&self, req: McpMessageRequest) -> impl Future<Output = Result<Value>> + Send {
        let inner = Arc::clone(&self.inner);
        async move { inner.mcp_message_dyn(req).await }
    }

    fn cancel_request(&self, req: ControlCancelRequest) -> impl Future<Output = ()> + Send {
        let inner = Arc::clone(&self.inner);
        async move { inner.cancel_request_dyn(req).await }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn noop_handler_denies_tools_and_acks_hooks() {
        let handler = AnyControlHandler::new(NoopControlHandler);
        let req = ToolPermissionRequest {
            tool_name: "Bash".to_string(),
            input: json!({"command": "ls"}),
            permission_suggestions: Vec::new(),
            blocked_path: None,
            decision_reason: None,
            tool_use_id: Some("tu-1".to_string()),
        };
        let decision = handler.can_use_tool(req).await.unwrap();
        assert!(matches!(
            decision,
            PermissionDecision::Deny {
                interrupt: false,
                ..
            }
        ));

        let hook_res = handler
            .hook_callback(HookCallbackRequest {
                callback_id: "cb-1".to_string(),
                input: json!({}),
                tool_use_id: None,
            })
            .await
            .unwrap();
        assert_eq!(hook_res, json!({}));
    }
}
