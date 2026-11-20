//! Conversions between ACP domain types and `claude-acp-sdk` / `claude-codes`
//! wire models (MCP servers, prompt content blocks, tool calls/diffs/results,
//! `TodoWrite` plans, and persisted transcript replay).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use aither_acp::domain::{
    ContentBlock as AcpContentBlock, ContentChunk, Diff, DiffChange, DiffOperation, DiffPatch,
    DiffPathChange, Location, MessageId, Open, Patch, PlanEntry, PlanEntryPriority,
    PlanEntryStatus, ResourceContents, SessionUpdate, ToolCallContent, ToolCallStatus,
    ToolCallUpdate, ToolKind,
};
use aither_acp::{ContentBlock as DomainContentBlock, McpServer, ToolCallId};
use claude_acp_sdk::{
    ClaudeOutput, ContentBlock as SdkContentBlock, ImageBlock, ImageSource, ImageSourceType,
    McpHttpServerConfig, McpServerConfig, McpSseServerConfig, McpStdioServerConfig, MediaType,
    TextBlock, ToolInput, ToolResultBlock, ToolResultContent, TranscriptEntry, TranscriptRecord,
    claude_codes::TodoStatus,
};
use serde_json::Value;

use crate::error::{ClaudeAcpError, Result};

/// Maximum characters included in a synthesized command/search tool call title.
const MAX_TOOL_TITLE_CHARS: usize = 72;

/// Plan id used for `TodoWrite`-derived plans (each write replaces the list).
const TODO_PLAN_ID: &str = "todos";

/// Convert ACP [`McpServer`] definitions from session setup params into
/// deterministic [`McpServerConfig`] entries for the SDK.
#[must_use]
pub fn convert_mcp_servers(servers: &[McpServer]) -> BTreeMap<String, McpServerConfig> {
    let mut out = BTreeMap::new();
    for server in servers {
        match server {
            McpServer::Stdio(stdio) => {
                let args = (!stdio.args.is_empty()).then(|| stdio.args.clone());
                let env: HashMap<String, String> = stdio
                    .env
                    .iter()
                    .map(|var| (var.name.clone(), var.value.clone()))
                    .collect();
                out.insert(
                    stdio.name.clone(),
                    McpServerConfig::Stdio(McpStdioServerConfig {
                        command: stdio.command.clone(),
                        args,
                        env: (!env.is_empty()).then_some(env),
                    }),
                );
            }
            McpServer::Http(http) => {
                let headers: HashMap<String, String> = http
                    .headers
                    .iter()
                    .map(|hdr| (hdr.name.clone(), hdr.value.clone()))
                    .collect();
                out.insert(
                    http.name.clone(),
                    McpServerConfig::Http(McpHttpServerConfig {
                        url: http.url.clone(),
                        headers: (!headers.is_empty()).then_some(headers),
                    }),
                );
            }
            // The v1 codec cannot express `sse` in the domain union, so a
            // v1 client-sent `{"type":"sse"}` arrives preserved as `Unknown`.
            McpServer::Unknown(tag, value) if tag == "sse" => {
                let name = value
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let url = value.get("url").and_then(Value::as_str).unwrap_or_default();
                if name.is_empty() || url.is_empty() {
                    continue;
                }
                let headers: HashMap<String, String> = value
                    .get("headers")
                    .and_then(Value::as_array)
                    .map(|hdrs| {
                        hdrs.iter()
                            .filter_map(|h| {
                                Some((
                                    h.get("name").and_then(Value::as_str)?.to_string(),
                                    h.get("value").and_then(Value::as_str)?.to_string(),
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                out.insert(
                    name.to_string(),
                    McpServerConfig::Sse(McpSseServerConfig {
                        url: url.to_string(),
                        headers: (!headers.is_empty()).then_some(headers),
                    }),
                );
            }
            McpServer::Custom(..) | McpServer::Unknown(..) => {}
        }
    }
    out
}

/// Convert ACP prompt [`DomainContentBlock`]s into Claude user input [`SdkContentBlock`]s.
///
/// # Errors
///
/// Returns [`ClaudeAcpError::InvalidParams`] if an unsupported `Audio`/`Custom`/
/// `Unknown` block or an empty `Image` block (neither `data` nor `uri`) is
/// supplied.
pub fn convert_prompt_blocks(blocks: &[DomainContentBlock]) -> Result<Vec<SdkContentBlock>> {
    if blocks.is_empty() {
        return Err(ClaudeAcpError::InvalidParams(
            "prompt must contain at least one content block".to_string(),
        ));
    }

    let mut converted = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            DomainContentBlock::Text(text) => {
                converted.push(sdk_text(text.text.clone()));
            }
            DomainContentBlock::Image(img) => {
                converted.push(convert_image_block(img)?);
            }
            DomainContentBlock::Resource(res) => match &res.resource {
                ResourceContents::Text(text) => {
                    let header = text.mime_type.as_deref().map_or_else(
                        || format!("<context uri=\"{}\">", text.uri),
                        |mime| format!("<context uri=\"{}\" mimeType=\"{mime}\">", text.uri),
                    );
                    converted.push(sdk_text(format!("{header}\n{}\n</context>", text.text)));
                }
                ResourceContents::Blob(blob) => {
                    let mime = blob
                        .mime_type
                        .as_deref()
                        .unwrap_or("application/octet-stream");
                    if mime.starts_with("image/") {
                        converted.push(SdkContentBlock::Image(ImageBlock {
                            source: ImageSource {
                                source_type: ImageSourceType::Base64,
                                media_type: MediaType::from(mime),
                                data: blob.blob.clone(),
                            },
                        }));
                    } else {
                        converted.push(sdk_text(format!(
                            "[Embedded binary resource: {} ({mime})]",
                            blob.uri
                        )));
                    }
                }
            },
            DomainContentBlock::ResourceLink(link) => {
                let mut desc = format!("[Resource link: {} ({})]", link.name, link.uri);
                if let Some(details) = link.description.as_deref().filter(|s| !s.is_empty()) {
                    desc.push_str(" — ");
                    desc.push_str(details);
                }
                converted.push(sdk_text(desc));
            }
            DomainContentBlock::Audio(_) => {
                return Err(ClaudeAcpError::InvalidParams(
                    "audio content blocks are not supported".to_string(),
                ));
            }
            DomainContentBlock::Custom(..) | DomainContentBlock::Unknown(..) => {
                return Err(ClaudeAcpError::InvalidParams(
                    "unsupported prompt content block".to_string(),
                ));
            }
        }
    }

    Ok(converted)
}

fn convert_image_block(img: &aither_acp::domain::ImageContent) -> Result<SdkContentBlock> {
    if let Some(data) = img.data.as_deref().filter(|d| !d.is_empty()) {
        return Ok(SdkContentBlock::Image(ImageBlock {
            source: ImageSource {
                source_type: ImageSourceType::Base64,
                media_type: MediaType::from(img.mime_type.as_str()),
                data: data.to_string(),
            },
        }));
    }

    if let Some(uri) = img.uri.as_deref().filter(|u| !u.is_empty()) {
        if let Some((mime, b64)) = parse_data_uri(uri) {
            let effective_mime = if mime.is_empty() {
                img.mime_type.as_str()
            } else {
                mime
            };
            return Ok(SdkContentBlock::Image(ImageBlock {
                source: ImageSource {
                    source_type: ImageSourceType::Base64,
                    media_type: MediaType::from(effective_mime),
                    data: b64.to_string(),
                },
            }));
        }
        return Ok(sdk_text(format!("[Image: {uri} ({})]", img.mime_type)));
    }

    Err(ClaudeAcpError::InvalidParams(
        "image content block requires non-empty `data` or `uri`".to_string(),
    ))
}

fn parse_data_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(";base64,")?;
    Some((meta, payload))
}

const fn sdk_text(text: String) -> SdkContentBlock {
    SdkContentBlock::Text(TextBlock {
        text,
        citations: Vec::new(),
    })
}

/// Extract plan entries when a tool call targets `TodoWrite`.
#[must_use]
pub fn extract_todo_plan(tool_name: &str, input: &Value) -> Option<Vec<PlanEntry>> {
    if tool_name != "TodoWrite" {
        return None;
    }

    if let Ok(ToolInput::TodoWrite(todo_input)) = serde_json::from_value::<ToolInput>(input.clone())
    {
        let entries = todo_input
            .todos
            .into_iter()
            .map(|item| plan_entry(item.content, map_todo_status(&item.status)))
            .collect();
        return Some(entries);
    }

    // Fallback when `activeForm` is omitted by custom/older payloads.
    let todos = input.get("todos")?.as_array()?;
    let mut entries = Vec::with_capacity(todos.len());
    for entry in todos {
        let content = entry.get("content")?.as_str()?.to_string();
        let status = match entry
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("pending")
        {
            "in_progress" => PlanEntryStatus::InProgress,
            "completed" => PlanEntryStatus::Completed,
            _ => PlanEntryStatus::Pending,
        };
        entries.push(plan_entry(content, status));
    }
    Some(entries)
}

const fn plan_entry(content: String, status: PlanEntryStatus) -> PlanEntry {
    PlanEntry {
        content,
        status: Some(Open::Known(status)),
        priority: Some(Open::Known(PlanEntryPriority::Medium)),
        meta: None,
        extra: BTreeMap::new(),
    }
}

const fn map_todo_status(status: &TodoStatus) -> PlanEntryStatus {
    match status {
        TodoStatus::InProgress => PlanEntryStatus::InProgress,
        TodoStatus::Completed => PlanEntryStatus::Completed,
        TodoStatus::Pending | TodoStatus::Unknown(_) => PlanEntryStatus::Pending,
    }
}

/// Extracted presentation metadata for a Claude tool invocation.
#[derive(Debug, Clone)]
pub struct ToolDescriptor {
    pub(crate) kind: ToolKind,
    pub(crate) title: String,
    pub(crate) locations: Vec<Location>,
    pub(crate) content: Vec<ToolCallContent>,
}

/// Derive ACP [`ToolKind`], title, file locations, and `Edit`/`Write` [`Diff`]s
/// from a Claude tool name and JSON input object.
#[must_use]
pub fn describe_tool(name: &str, input: &Value) -> ToolDescriptor {
    if let Some(file_desc) = describe_file_tool(name, input) {
        return file_desc;
    }
    describe_action_tool(name, input)
}

fn describe_file_tool(name: &str, input: &Value) -> Option<ToolDescriptor> {
    match name {
        "Read" => {
            let path = str_field(input, "file_path");
            let line = input
                .get("offset")
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok());
            let title = path.map_or_else(|| "Read file".to_string(), |p| format!("Read {p}"));
            let locations = path
                .map(|p| vec![loc_with_line(p, line)])
                .unwrap_or_default();
            Some(ToolDescriptor {
                kind: ToolKind::Read,
                title,
                locations,
                content: Vec::new(),
            })
        }
        "NotebookRead" => {
            let path = str_field(input, "notebook_path");
            Some(ToolDescriptor {
                kind: ToolKind::Read,
                title: path.map_or_else(|| "Read notebook".to_string(), |p| format!("Read {p}")),
                locations: path.map(loc_only).unwrap_or_default(),
                content: Vec::new(),
            })
        }
        "Edit" => Some(describe_single_diff_tool(
            input,
            "file_path",
            Some("old_string"),
            "new_string",
            "Edit",
        )),
        "Write" => Some(describe_single_diff_tool(
            input,
            "file_path",
            None,
            "content",
            "Write",
        )),
        "NotebookEdit" => Some(describe_single_diff_tool(
            input,
            "notebook_path",
            None,
            "new_source",
            "Edit",
        )),
        "MultiEdit" => {
            let path = str_field(input, "file_path");
            let mut content = Vec::new();
            if let (Some(p), Some(edits)) = (path, input.get("edits").and_then(Value::as_array)) {
                for op in edits {
                    if let Some(new_s) = str_field(op, "new_string") {
                        content.push(ToolCallContent::Diff(diff_modify(
                            p,
                            str_field(op, "old_string"),
                            new_s,
                        )));
                    }
                }
            }
            Some(ToolDescriptor {
                kind: ToolKind::Edit,
                title: path.map_or_else(|| "Edit file".to_string(), |p| format!("Edit {p}")),
                locations: path.map(loc_only).unwrap_or_default(),
                content,
            })
        }
        _ => None,
    }
}

fn describe_single_diff_tool(
    input: &Value,
    path_key: &str,
    old_key: Option<&str>,
    new_key: &str,
    verb: &str,
) -> ToolDescriptor {
    let path = str_field(input, path_key);
    let old_text = old_key.and_then(|k| str_field(input, k));
    let new_text = str_field(input, new_key);
    let content = match (path, new_text) {
        (Some(p), Some(new_s)) => vec![ToolCallContent::Diff(diff_modify(p, old_text, new_s))],
        _ => Vec::new(),
    };
    ToolDescriptor {
        kind: ToolKind::Edit,
        title: path.map_or_else(|| format!("{verb} file"), |p| format!("{verb} {p}")),
        locations: path.map(loc_only).unwrap_or_default(),
        content,
    }
}

/// Build a [`Diff`] describing a modify-in-place with a synthesized git patch.
fn diff_modify(path: &str, old_text: Option<&str>, new_text: &str) -> Diff {
    let mut patch_text = format!("--- a/{path}\n+++ b/{path}\n");
    if let Some(old) = old_text {
        for line in old.lines() {
            let _ = writeln!(patch_text, "-{line}");
        }
    }
    for line in new_text.lines() {
        let _ = writeln!(patch_text, "+{line}");
    }
    Diff {
        changes: vec![DiffChange {
            operation: Open::Known(DiffOperation::Modify(DiffPathChange {
                path: PathBuf::from(path),
            })),
            file_type: None,
            mime_type: None,
            meta: None,
            extra: BTreeMap::new(),
        }],
        patch: Some(DiffPatch {
            format: Open::Known(aither_acp::domain::DiffPatchFormat::GitPatch),
            text: patch_text,
            meta: None,
        }),
        meta: None,
        extra: BTreeMap::new(),
    }
}

fn describe_action_tool(name: &str, input: &Value) -> ToolDescriptor {
    match name {
        "Bash" => {
            let title = str_field(input, "description")
                .map(truncate_title)
                .or_else(|| {
                    str_field(input, "command").map(|cmd| truncate_title(&format!("`{cmd}`")))
                })
                .unwrap_or_else(|| "Bash".to_string());
            ToolDescriptor {
                kind: ToolKind::Execute,
                title,
                locations: Vec::new(),
                content: Vec::new(),
            }
        }
        "KillShell" => ToolDescriptor {
            kind: ToolKind::Execute,
            title: str_field(input, "shell_id")
                .map_or_else(|| "Kill shell".to_string(), |id| format!("Kill shell {id}")),
            locations: Vec::new(),
            content: Vec::new(),
        },
        "Glob" | "Grep" => {
            let pattern = str_field(input, "pattern");
            let path = str_field(input, "path");
            ToolDescriptor {
                kind: ToolKind::Search,
                title: pattern.map_or_else(
                    || name.to_string(),
                    |pat| truncate_title(&format!("{name} `{pat}`")),
                ),
                locations: path.map(loc_only).unwrap_or_default(),
                content: Vec::new(),
            }
        }
        "LS" => {
            let path = str_field(input, "path");
            ToolDescriptor {
                kind: ToolKind::Search,
                title: path.map_or_else(|| "List directory".to_string(), |p| format!("List {p}")),
                locations: path.map(loc_only).unwrap_or_default(),
                content: Vec::new(),
            }
        }
        "WebSearch" | "ToolSearch" => ToolDescriptor {
            kind: ToolKind::Search,
            title: str_field(input, "query").map_or_else(
                || name.to_string(),
                |q| truncate_title(&format!("{name}: {q}")),
            ),
            locations: Vec::new(),
            content: Vec::new(),
        },
        "WebFetch" => ToolDescriptor {
            kind: ToolKind::Fetch,
            title: str_field(input, "url").map_or_else(
                || "WebFetch".to_string(),
                |u| truncate_title(&format!("Fetch {u}")),
            ),
            locations: Vec::new(),
            content: Vec::new(),
        },
        "TodoWrite" | "Task" => ToolDescriptor {
            kind: ToolKind::Think,
            title: str_field(input, "description").map_or_else(|| name.to_string(), truncate_title),
            locations: Vec::new(),
            content: Vec::new(),
        },
        "TaskOutput" => ToolDescriptor {
            kind: ToolKind::Read,
            title: str_field(input, "task_id")
                .map_or_else(|| "TaskOutput".to_string(), |id| format!("Read task {id}")),
            locations: Vec::new(),
            content: Vec::new(),
        },
        "EnterPlanMode" | "ExitPlanMode" => ToolDescriptor {
            kind: ToolKind::SwitchMode,
            title: name.to_string(),
            locations: Vec::new(),
            content: Vec::new(),
        },
        _ => ToolDescriptor {
            kind: ToolKind::Other,
            title: str_field(input, "description").map_or_else(|| name.to_string(), truncate_title),
            locations: Vec::new(),
            content: Vec::new(),
        },
    }
}

/// Construct a full [`ToolCallUpdate`] (all fields set) for a new tool call
/// with id `tool_call_id` (the upstream `tool_use_id`).
#[must_use]
pub fn build_tool_call(
    tool_call_id: &str,
    name: &str,
    input: &Value,
    status: ToolCallStatus,
) -> ToolCallUpdate {
    let desc = describe_tool(name, input);
    ToolCallUpdate {
        tool_call_id: ToolCallId::from_string(tool_call_id),
        name: Patch::Set(name.to_string()),
        title: Patch::Set(desc.title),
        kind: Patch::Set(Open::Known(desc.kind)),
        status: Patch::Set(Open::Known(status)),
        content: Patch::Set(desc.content),
        locations: Patch::Set(desc.locations),
        raw_input: if input.is_null() {
            Patch::Unchanged
        } else {
            Patch::Set(input.clone())
        },
        raw_output: Patch::Unchanged,
        capabilities: Patch::Unchanged,
        meta: Patch::Unchanged,
        extra: BTreeMap::new(),
    }
}

/// Build a [`ToolCallUpdate`] enriching an earlier announcement once the full
/// JSON input has arrived (only non-default fields patched).
#[must_use]
pub fn build_tool_input_update(
    tool_call_id: &str,
    name: &str,
    input: &Value,
    status: ToolCallStatus,
) -> ToolCallUpdate {
    let desc = describe_tool(name, input);
    ToolCallUpdate {
        tool_call_id: ToolCallId::from_string(tool_call_id),
        status: Patch::Set(Open::Known(status)),
        content: if desc.content.is_empty() {
            Patch::Unchanged
        } else {
            Patch::Set(desc.content)
        },
        title: Patch::Set(desc.title),
        kind: Patch::Set(Open::Known(desc.kind)),
        locations: if desc.locations.is_empty() {
            Patch::Unchanged
        } else {
            Patch::Set(desc.locations)
        },
        raw_input: Patch::Set(input.clone()),
        ..ToolCallUpdate::default()
    }
}

/// Convert a CLI [`ToolResultBlock`] (plus optional structured `tool_use_result`)
/// into a terminal [`ToolCallUpdate`].
#[must_use]
pub fn build_tool_result_update(
    tr: &ToolResultBlock,
    tool_use_result: Option<&Value>,
    preserve_diff: Option<&[ToolCallContent]>,
) -> ToolCallUpdate {
    let status = if tr.is_error == Some(true) {
        ToolCallStatus::Failed
    } else {
        ToolCallStatus::Completed
    };

    let mut content_blocks: Vec<ToolCallContent> = Vec::new();
    if let Some(existing_diffs) = preserve_diff {
        content_blocks.extend(existing_diffs.iter().cloned());
    }

    if let Some(text) = extract_tool_result_text(tr, tool_use_result)
        && !text.is_empty()
    {
        content_blocks.push(ToolCallContent::Content(
            aither_acp::domain::tool::Content {
                content: AcpContentBlock::text(text),
                meta: None,
                extra: BTreeMap::new(),
            },
        ));
    }

    let raw_output = tool_use_result.cloned().or_else(|| {
        tr.content
            .as_ref()
            .and_then(|c| serde_json::to_value(c).ok())
    });

    ToolCallUpdate {
        tool_call_id: ToolCallId::from_string(tr.tool_use_id.clone()),
        status: Patch::Set(Open::Known(status)),
        content: if content_blocks.is_empty() {
            Patch::Unchanged
        } else {
            Patch::Set(content_blocks)
        },
        raw_output: raw_output.map_or(Patch::Unchanged, Patch::Set),
        ..ToolCallUpdate::default()
    }
}

fn extract_tool_result_text(
    tr: &ToolResultBlock,
    tool_use_result: Option<&Value>,
) -> Option<String> {
    if let Some(content) = &tr.content {
        match content {
            ToolResultContent::Text(text) => return Some(text.clone()),
            ToolResultContent::Structured(items) => {
                let mut parts = Vec::new();
                for item in items {
                    if let Some(t) = item.get("text").and_then(Value::as_str) {
                        parts.push(t.to_string());
                    } else if let Ok(serialized) = serde_json::to_string(item) {
                        parts.push(serialized);
                    }
                }
                if !parts.is_empty() {
                    return Some(parts.join("\n"));
                }
            }
        }
    }

    let obj = tool_use_result?.as_object()?;
    let stdout = obj
        .get("stdout")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let stderr = obj
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => None,
        (false, true) => Some(stdout.to_string()),
        (true, false) => Some(stderr.to_string()),
        (false, false) => Some(format!("{stdout}\n{stderr}")),
    }
}

/// Convert persisted [`TranscriptEntry`] records into chronological ACP
/// [`SessionUpdate`]s for `session/resume` replay.
#[must_use]
pub fn transcript_to_updates(entries: &[TranscriptEntry]) -> Vec<SessionUpdate> {
    let mut updates = Vec::new();
    let mut todo_ids: HashSet<String> = HashSet::new();
    let mut tool_diffs: HashMap<String, Vec<ToolCallContent>> = HashMap::new();

    for entry in entries {
        if entry.is_sidechain || entry.is_meta {
            continue;
        }
        let TranscriptRecord::Output(output) = &entry.record else {
            continue;
        };

        match output {
            ClaudeOutput::User(user) => {
                replay_user_entry(entry, user, &todo_ids, &tool_diffs, &mut updates);
            }
            ClaudeOutput::Assistant(assistant) => {
                replay_assistant_entry(assistant, &mut todo_ids, &mut tool_diffs, &mut updates);
            }
            _ => {}
        }
    }

    updates
}

fn replay_user_entry(
    entry: &TranscriptEntry,
    user: &claude_acp_sdk::UserMessage,
    todo_ids: &HashSet<String>,
    tool_diffs: &HashMap<String, Vec<ToolCallContent>>,
    updates: &mut Vec<SessionUpdate>,
) {
    if user.parent_tool_use_id.is_some() || user.agent_id.is_some() {
        return;
    }
    let mut had_tool_result = false;
    for block in &user.message.content {
        if let SdkContentBlock::ToolResult(tr) = block {
            had_tool_result = true;
            if todo_ids.contains(&tr.tool_use_id) {
                continue;
            }
            let diffs = tool_diffs.get(&tr.tool_use_id).map(Vec::as_slice);
            updates.push(SessionUpdate::ToolCallUpdate(build_tool_result_update(
                tr,
                user.tool_use_result.as_ref(),
                diffs,
            )));
        }
    }

    if !had_tool_result && user.tool_use_result.is_none() {
        let msg_id = user
            .uuid
            .clone()
            .or_else(|| entry.uuid.clone())
            .map_or_else(MessageId::new, MessageId::from_string);
        for block in &user.message.content {
            if let Some(acp_block) = sdk_block_to_acp(block) {
                updates.push(SessionUpdate::UserMessageChunk(ContentChunk {
                    content: acp_block,
                    message_id: msg_id.clone(),
                    meta: None,
                    extra: BTreeMap::new(),
                }));
            }
        }
    }
}

fn replay_assistant_entry(
    assistant: &claude_acp_sdk::AssistantMessage,
    todo_ids: &mut HashSet<String>,
    tool_diffs: &mut HashMap<String, Vec<ToolCallContent>>,
    updates: &mut Vec<SessionUpdate>,
) {
    if assistant.parent_tool_use_id.is_some() || assistant.agent_id.is_some() {
        return;
    }
    let msg_id = (!assistant.message.id.is_empty())
        .then(|| assistant.message.id.clone())
        .or_else(|| assistant.uuid.clone())
        .map_or_else(MessageId::new, MessageId::from_string);

    for block in &assistant.message.content {
        match block {
            SdkContentBlock::Thinking(tb) if !tb.thinking.is_empty() => {
                updates.push(SessionUpdate::AgentThoughtChunk(ContentChunk {
                    content: AcpContentBlock::text(tb.thinking.clone()),
                    message_id: msg_id.clone(),
                    meta: None,
                    extra: BTreeMap::new(),
                }));
            }
            SdkContentBlock::Text(tb) if !tb.text.is_empty() => {
                updates.push(SessionUpdate::AgentMessageChunk(ContentChunk {
                    content: AcpContentBlock::text(tb.text.clone()),
                    message_id: msg_id.clone(),
                    meta: None,
                    extra: BTreeMap::new(),
                }));
            }
            SdkContentBlock::ToolUse(tu) => {
                if let Some(entries) = extract_todo_plan(&tu.name, &tu.input) {
                    todo_ids.insert(tu.id.clone());
                    updates.push(plan_update(entries));
                } else {
                    let call =
                        build_tool_call(&tu.id, &tu.name, &tu.input, ToolCallStatus::InProgress);
                    if let Patch::Set(content) = &call.content
                        && !content.is_empty()
                    {
                        tool_diffs.insert(tu.id.clone(), content.clone());
                    }
                    updates.push(SessionUpdate::ToolCallUpdate(call));
                }
            }
            SdkContentBlock::McpToolUse(mcp) => {
                updates.push(SessionUpdate::ToolCallUpdate(build_tool_call(
                    &mcp.id,
                    &mcp.name,
                    &mcp.input,
                    ToolCallStatus::InProgress,
                )));
            }
            SdkContentBlock::ServerToolUse(srv) => {
                updates.push(SessionUpdate::ToolCallUpdate(build_tool_call(
                    &srv.id,
                    &srv.name,
                    &srv.input,
                    ToolCallStatus::InProgress,
                )));
            }
            _ => {}
        }
    }
}

/// Wrap plan entries into a [`SessionUpdate::PlanUpdate`].
#[must_use]
pub fn plan_update(entries: Vec<PlanEntry>) -> SessionUpdate {
    SessionUpdate::PlanUpdate(aither_acp::domain::PlanUpdate {
        plan: Open::Known(aither_acp::domain::Plan::Items(
            aither_acp::domain::PlanItems {
                plan_id: aither_acp::PlanId::from_string(TODO_PLAN_ID),
                entries,
                meta: None,
                extra: BTreeMap::new(),
            },
        )),
        meta: None,
        extra: BTreeMap::new(),
    })
}

fn sdk_block_to_acp(block: &SdkContentBlock) -> Option<AcpContentBlock> {
    match block {
        SdkContentBlock::Text(tb) if !tb.text.is_empty() => {
            Some(AcpContentBlock::text(tb.text.clone()))
        }
        SdkContentBlock::Image(img) => {
            Some(AcpContentBlock::Image(aither_acp::domain::ImageContent {
                data: Some(img.source.data.clone()),
                uri: None,
                mime_type: img.source.media_type.as_str().to_string(),
                annotations: None,
                meta: None,
                extra: BTreeMap::new(),
            }))
        }
        _ => None,
    }
}

fn str_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn loc_only(path: &str) -> Vec<Location> {
    vec![loc_with_line(path, None)]
}

fn loc_with_line(path: &str, line: Option<u32>) -> Location {
    Location {
        path: PathBuf::from(path),
        line,
        meta: None,
    }
}

fn truncate_title(text: &str) -> String {
    let single_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = single_line.chars();
    let prefix: String = chars.by_ref().take(MAX_TOOL_TITLE_CHARS).collect();
    if chars.next().is_some() {
        format!("{prefix}…")
    } else {
        prefix
    }
}
