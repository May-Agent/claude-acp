//! Non-blocking helpers for listing, replaying, and deleting persisted Claude
//! Code session transcripts (`~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`).

use std::cmp::Reverse;
use std::future::{Future, poll_fn};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::task::Poll;

use chrono::{DateTime, SecondsFormat, Utc};
use claude_codes::ClaudeOutput;
pub use claude_codes::transcript::{encode_project_dir, transcript_path};
use futures_lite::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, SdkError};

/// Maximum concurrent filesystem transcript reads when scanning project folders,
/// bounding simultaneous open file descriptors on large transcript stores.
const MAX_CONCURRENT_TRANSCRIPT_READS: usize = 16;

/// Maximum character length when deriving a fallback session title from the
/// first user prompt.
const MAX_PROMPT_TITLE_CHARS: usize = 120;

/// Handle to a Claude Code transcript store rooted at `<home>/.claude/projects`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptStore {
    home: PathBuf,
}

impl TranscriptStore {
    /// Construct a [`TranscriptStore`] for an explicit home directory
    /// (pass a temporary directory in tests).
    #[must_use]
    pub fn new(home: impl Into<PathBuf>) -> Self {
        Self { home: home.into() }
    }

    /// Construct a [`TranscriptStore`] using `HOME` (or `USERPROFILE` on Windows)
    /// from the current process environment.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Io`] if no home directory environment variable is set.
    pub fn from_env() -> Result<Self> {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .filter(|val| !val.is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| {
                std::io::Error::new(
                    ErrorKind::NotFound,
                    "unable to determine home directory from HOME or USERPROFILE",
                )
            })?;
        Ok(Self { home })
    }

    /// Root home directory backing this store.
    #[must_use]
    pub fn home(&self) -> &Path {
        &self.home
    }

    /// Path to `<home>/.claude/projects`.
    #[must_use]
    pub fn projects_dir(&self) -> PathBuf {
        self.home.join(".claude").join("projects")
    }

    /// List persisted sessions, optionally filtered to `cwd`, sorted by
    /// `updated_at` descending.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Io`] if reading an existing directory fails.
    pub async fn list_sessions(&self, cwd: Option<&Path>) -> Result<Vec<SessionSummary>> {
        list_sessions(&self.home, cwd).await
    }

    /// Load all transcript records for `session_id` (optionally scoped to `cwd`
    /// first) in chronological file order for `session/load` replay.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Protocol`] if `session_id` is invalid or a line is
    /// malformed JSON, or [`SdkError::Io`] if the transcript file does not exist.
    pub async fn load_transcript(
        &self,
        session_id: &str,
        cwd: Option<&Path>,
    ) -> Result<Vec<TranscriptEntry>> {
        load_transcript(&self.home, session_id, cwd).await
    }

    /// Delete the `.jsonl` transcript file for `session_id`, returning `Ok(true)`
    /// if a file was removed or `Ok(false)` if no matching transcript existed.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Protocol`] if `session_id` is invalid or
    /// [`SdkError::Io`] on filesystem permission errors.
    pub async fn delete_transcript(&self, session_id: &str, cwd: Option<&Path>) -> Result<bool> {
        delete_transcript(&self.home, session_id, cwd).await
    }
}

/// Summary metadata for a persisted Claude Code session discovered on disk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSummary {
    /// Session identifier (matches the `<session-id>.jsonl` filename stem).
    pub session_id: String,
    /// Working directory recorded inside the transcript, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Human-readable session title (`custom-title` / `ai-title` / `summary`,
    /// falling back to the initial user prompt).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// ISO 8601 timestamp of the session's most recent activity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<String>,
    /// Path to the backing `.jsonl` file on disk.
    pub path: PathBuf,
}

/// One parsed line from a persisted session `.jsonl` transcript file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    /// 1-based line number within the `.jsonl` file.
    pub line: usize,
    /// Envelope record UUID, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uuid: Option<String>,
    /// Parent record UUID in the conversation tree, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_uuid: Option<String>,
    /// ISO 8601 timestamp recorded on this line, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Working directory recorded on this line, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// True when this line belongs to a subagent sidechain (`isSidechain`).
    pub is_sidechain: bool,
    /// True when this line carries internal/synthetic metadata (`isMeta`,
    /// `isCompactSummary`, or `isVisibleInTranscriptOnly`).
    pub is_meta: bool,
    /// Typed or preserved transcript payload.
    pub record: TranscriptRecord,
}

/// Decoded payload of a [`TranscriptEntry`].
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TranscriptRecord {
    /// Typed `claude-codes` protocol/transcript message (`User`, `Assistant`,
    /// `System`, `Result`, `AiTitle`, `Summary`, ...).
    Output(ClaudeOutput),
    /// Forward-compatible fallback for unrecognized CLI journal record types.
    Unknown {
        /// Value of the `"type"` field on the JSON object.
        record_type: String,
        /// Full JSON object preserved verbatim.
        raw: Value,
    },
}

/// List persisted Claude sessions under `<home>/.claude/projects`, optionally
/// filtered to `cwd`, sorted by `updated_at` descending.
///
/// Because [`encode_project_dir`] is lossy (`/a/b.c` and `/a/b/c` map to the
/// same folder name), filtering by `cwd` also verifies the unencoded `"cwd"`
/// recorded inside each transcript whenever present.
///
/// # Errors
///
/// Returns [`SdkError::Io`] if scanning an existing project directory fails.
pub async fn list_sessions(home: &Path, cwd: Option<&Path>) -> Result<Vec<SessionSummary>> {
    let projects_root = home.join(".claude").join("projects");
    let candidate_files = match cwd {
        Some(target_cwd) => {
            let project_dir = projects_root.join(encode_project_dir(target_cwd));
            collect_jsonl_in_dir(&project_dir).await?
        }
        None => collect_all_project_jsonls(&projects_root).await?,
    };

    let inspected = map_bounded(
        candidate_files,
        MAX_CONCURRENT_TRANSCRIPT_READS,
        inspect_transcript_file,
    )
    .await;

    let mut ranked: Vec<(Option<DateTime<Utc>>, SessionSummary)> = Vec::new();
    for item in inspected {
        let Some((sort_key, mut summary)) = item? else {
            continue;
        };
        if let Some(target_cwd) = cwd {
            match &summary.cwd {
                Some(recorded) if recorded != target_cwd => continue,
                None => summary.cwd = Some(target_cwd.to_path_buf()),
                Some(_) => {}
            }
        }
        ranked.push((sort_key, summary));
    }

    ranked.sort_by(|(ts_a, sum_a), (ts_b, sum_b)| {
        Reverse(*ts_a)
            .cmp(&Reverse(*ts_b))
            .then_with(|| sum_a.session_id.cmp(&sum_b.session_id))
    });

    Ok(ranked.into_iter().map(|(_, summary)| summary).collect())
}

/// Locate the `.jsonl` file path for `session_id` under `<home>/.claude/projects`.
///
/// Checks `transcript_path(home, cwd, session_id)` first when `cwd` is supplied,
/// then scans project directories if `cwd` was omitted (e.g. `session/delete`).
///
/// # Errors
///
/// Returns [`SdkError::Protocol`] if `session_id` contains path separators or
/// traversal segments, or [`SdkError::Io`] on directory read failure.
pub async fn find_transcript_path(
    home: &Path,
    session_id: &str,
    cwd: Option<&Path>,
) -> Result<Option<PathBuf>> {
    validate_session_id(session_id)?;

    if let Some(dir) = cwd {
        let direct = transcript_path(home, dir, session_id);
        if file_exists(&direct).await? {
            return Ok(Some(direct));
        }
    }

    let projects_root = home.join(".claude").join("projects");
    let mut entries = match async_fs::read_dir(&projects_root).await {
        Ok(dir) => dir,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(SdkError::from(err)),
    };

    let file_name = format!("{session_id}.jsonl");
    let mut candidate_paths = Vec::new();
    while let Some(entry) = entries.try_next().await? {
        let project_path = entry.path();
        candidate_paths.push(project_path.join(&file_name));
    }

    let checks = map_bounded(
        candidate_paths,
        MAX_CONCURRENT_TRANSCRIPT_READS,
        |candidate| async move {
            match file_exists(&candidate).await {
                Ok(true) => Ok(Some(candidate)),
                Ok(false) => Ok(None),
                Err(err) => Err(err),
            }
        },
    )
    .await;

    for check in checks {
        if let Some(found) = check? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// Load and parse all records from `session_id`'s transcript file for
/// `session/load` history replay.
///
/// # Errors
///
/// Returns [`SdkError::Protocol`] if `session_id` is invalid or a JSONL line is
/// malformed, or [`SdkError::Io`] (`ErrorKind::NotFound`) if the transcript does
/// not exist.
pub async fn load_transcript(
    home: &Path,
    session_id: &str,
    cwd: Option<&Path>,
) -> Result<Vec<TranscriptEntry>> {
    let path = find_transcript_path(home, session_id, cwd)
        .await?
        .ok_or_else(|| {
            SdkError::from(std::io::Error::new(
                ErrorKind::NotFound,
                format!("transcript for session `{session_id}` not found"),
            ))
        })?;

    let content = async_fs::read_to_string(&path).await?;
    parse_transcript_lines(&content)
}

/// Delete the persisted `.jsonl` transcript file for `session_id`.
///
/// Returns `Ok(true)` if the file was deleted or `Ok(false)` if it did not exist.
///
/// # Errors
///
/// Returns [`SdkError::Protocol`] if `session_id` is invalid or [`SdkError::Io`]
/// if removing the file fails for a reason other than `NotFound`.
pub async fn delete_transcript(home: &Path, session_id: &str, cwd: Option<&Path>) -> Result<bool> {
    let Some(path) = find_transcript_path(home, session_id, cwd).await? else {
        return Ok(false);
    };
    match async_fs::remove_file(&path).await {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(SdkError::from(err)),
    }
}

fn validate_session_id(session_id: &str) -> Result<()> {
    if session_id.is_empty()
        || session_id == "."
        || session_id == ".."
        || session_id.contains('/')
        || session_id.contains('\\')
        || session_id.contains('\0')
    {
        return Err(SdkError::Protocol(format!(
            "invalid session_id `{session_id}`"
        )));
    }
    Ok(())
}

async fn file_exists(path: &Path) -> Result<bool> {
    match async_fs::metadata(path).await {
        Ok(meta) => Ok(meta.is_file()),
        Err(err) if err.kind() == ErrorKind::NotFound => Ok(false),
        Err(err) => Err(SdkError::from(err)),
    }
}

async fn collect_jsonl_in_dir(project_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = match async_fs::read_dir(project_dir).await {
        Ok(dir) => dir,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(SdkError::from(err)),
    };
    let mut files = Vec::new();
    while let Some(entry) = entries.try_next().await? {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    Ok(files)
}

async fn collect_all_project_jsonls(projects_root: &Path) -> Result<Vec<PathBuf>> {
    let mut root_entries = match async_fs::read_dir(projects_root).await {
        Ok(dir) => dir,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(SdkError::from(err)),
    };
    let mut project_dirs = Vec::new();
    while let Some(entry) = root_entries.try_next().await? {
        let file_type = entry.file_type().await?;
        if file_type.is_dir() {
            project_dirs.push(entry.path());
        }
    }

    let nested = map_bounded(
        project_dirs,
        MAX_CONCURRENT_TRANSCRIPT_READS,
        |dir| async move { collect_jsonl_in_dir(&dir).await },
    )
    .await;

    let mut all_files = Vec::new();
    for dir_result in nested {
        all_files.extend(dir_result?);
    }
    Ok(all_files)
}

struct ScannedMetadata {
    cwd: Option<PathBuf>,
    title: Option<String>,
    best_ts: Option<(DateTime<Utc>, String)>,
}

fn scan_transcript_metadata(content: &str) -> ScannedMetadata {
    let mut cwd: Option<PathBuf> = None;
    let mut explicit_title: Option<String> = None;
    let mut first_user_prompt: Option<String> = None;
    let mut last_prompt_fallback: Option<String> = None;
    let mut best_ts: Option<(DateTime<Utc>, String)> = None;

    for raw_line in content.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.as_bytes().iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(obj) = value.as_object() else {
            continue;
        };

        if cwd.is_none()
            && let Some(dir_str) = obj.get("cwd").and_then(Value::as_str)
            && !dir_str.is_empty()
        {
            cwd = Some(PathBuf::from(dir_str));
        }

        if let Some(ts_str) = obj.get("timestamp").and_then(Value::as_str)
            && let Ok(parsed) = DateTime::parse_from_rfc3339(ts_str)
        {
            let parsed_utc = parsed.with_timezone(&Utc);
            if best_ts.as_ref().is_none_or(|(prev, _)| parsed_utc >= *prev) {
                best_ts = Some((parsed_utc, ts_str.to_string()));
            }
        }

        update_titles(
            obj,
            &mut explicit_title,
            &mut first_user_prompt,
            &mut last_prompt_fallback,
        );
    }

    ScannedMetadata {
        cwd,
        title: explicit_title
            .or(first_user_prompt)
            .or(last_prompt_fallback),
        best_ts,
    }
}

fn update_titles(
    obj: &serde_json::Map<String, Value>,
    explicit_title: &mut Option<String>,
    first_user_prompt: &mut Option<String>,
    last_prompt_fallback: &mut Option<String>,
) {
    let record_type = obj.get("type").and_then(Value::as_str).unwrap_or_default();
    match record_type {
        "custom-title" => {
            if let Some(t) = non_empty_field(obj, "customTitle") {
                *explicit_title = Some(t);
            }
        }
        "ai-title" => {
            if let Some(t) = non_empty_field(obj, "aiTitle") {
                *explicit_title = Some(t);
            }
        }
        "summary" => {
            if let Some(t) = non_empty_field(obj, "summary") {
                *explicit_title = Some(t);
            }
        }
        "system" if obj.get("subtype").and_then(Value::as_str) == Some("session_title_changed") => {
            if let Some(t) = non_empty_field(obj, "title") {
                *explicit_title = Some(t);
            }
        }
        "last-prompt" => {
            if let Some(p) = non_empty_field(obj, "lastPrompt") {
                *last_prompt_fallback = Some(truncate_title(&p));
            }
        }
        "user" if first_user_prompt.is_none() && !is_meta_or_sidechain(obj) => {
            if obj.get("tool_use_result").is_none()
                && let Some(text) = extract_user_prompt_text(obj)
            {
                *first_user_prompt = Some(truncate_title(&text));
            }
        }
        _ => {}
    }
}

#[allow(clippy::type_complexity)]
async fn inspect_transcript_file(
    path: PathBuf,
) -> Result<Option<(Option<DateTime<Utc>>, SessionSummary)>> {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return Ok(None);
    };
    let session_id = stem.to_string();

    let (content_res, meta_res) =
        futures_lite::future::zip(async_fs::read_to_string(&path), async_fs::metadata(&path)).await;

    let content = match content_res {
        Ok(text) => text,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(SdkError::from(err)),
    };
    let mtime = meta_res.ok().and_then(|m| m.modified().ok());
    let scanned = scan_transcript_metadata(&content);

    let (sort_key, updated_at) = match (scanned.best_ts, mtime) {
        (Some((parsed, raw_str)), _) => (Some(parsed), Some(raw_str)),
        (None, Some(sys_time)) => {
            let dt = DateTime::<Utc>::from(sys_time);
            let formatted = dt.to_rfc3339_opts(SecondsFormat::Millis, true);
            (Some(dt), Some(formatted))
        }
        (None, None) => (None, None),
    };

    Ok(Some((
        sort_key,
        SessionSummary {
            session_id,
            cwd: scanned.cwd,
            title: scanned.title,
            updated_at,
            path,
        },
    )))
}

fn parse_transcript_lines(content: &str) -> Result<Vec<TranscriptEntry>> {
    let mut entries = Vec::new();
    for (idx, raw_line) in content.split('\n').enumerate() {
        let line_no = idx + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.as_bytes().iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|err| {
            SdkError::Protocol(format!(
                "invalid JSON in transcript at line {line_no}: {err}"
            ))
        })?;
        let Some(obj) = value.as_object() else {
            return Err(SdkError::Protocol(format!(
                "transcript line {line_no} is not a JSON object"
            )));
        };
        let Some(record_type) = obj
            .get("type")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
        else {
            return Err(SdkError::Protocol(format!(
                "transcript line {line_no} missing string `type` field"
            )));
        };

        let uuid = obj
            .get("uuid")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let parent_uuid = obj
            .get("parentUuid")
            .or_else(|| obj.get("parent_uuid"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let timestamp = obj
            .get("timestamp")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let cwd = obj
            .get("cwd")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        let is_sidechain = obj
            .get("isSidechain")
            .or_else(|| obj.get("is_sidechain"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let is_meta = is_meta_only(obj);

        let record = serde_json::from_value::<ClaudeOutput>(value.clone()).map_or(
            TranscriptRecord::Unknown {
                record_type,
                raw: value,
            },
            TranscriptRecord::Output,
        );

        entries.push(TranscriptEntry {
            line: line_no,
            uuid,
            parent_uuid,
            timestamp,
            cwd,
            is_sidechain,
            is_meta,
            record,
        });
    }
    Ok(entries)
}

fn non_empty_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    let val = obj.get(key)?.as_str()?.trim();
    if val.is_empty() {
        None
    } else {
        Some(val.to_string())
    }
}

fn is_meta_only(obj: &serde_json::Map<String, Value>) -> bool {
    let flag = |camel: &str, snake: &str| {
        obj.get(camel)
            .or_else(|| obj.get(snake))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    flag("isMeta", "is_meta")
        || flag("isCompactSummary", "is_compact_summary")
        || flag("isVisibleInTranscriptOnly", "is_visible_in_transcript_only")
        || flag("isSynthetic", "is_synthetic")
}

fn is_meta_or_sidechain(obj: &serde_json::Map<String, Value>) -> bool {
    is_meta_only(obj)
        || obj
            .get("isSidechain")
            .or_else(|| obj.get("is_sidechain"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
}

fn extract_user_prompt_text(obj: &serde_json::Map<String, Value>) -> Option<String> {
    let content = obj.get("message")?.get("content")?;
    match content {
        Value::String(s) => {
            let trimmed = s.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_string())
        }
        Value::Array(blocks) => {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = block.get("text").and_then(Value::as_str)
                {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn truncate_title(raw: &str) -> String {
    let collapsed: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = collapsed.chars();
    let head: String = chars.by_ref().take(MAX_PROMPT_TITLE_CHARS).collect();
    if chars.next().is_some() {
        format!("{head}…")
    } else {
        head
    }
}

async fn map_bounded<I, T, R, F, Fut>(items: I, bound: usize, mut f: F) -> Vec<R>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T) -> Fut,
    Fut: Future<Output = R> + Send,
{
    let mut iter = items.into_iter();
    let limit = bound.max(1);
    let mut active: Vec<Pin<Box<dyn Future<Output = R> + Send>>> = Vec::with_capacity(limit);
    let mut results = Vec::new();

    for item in iter.by_ref().take(limit) {
        active.push(Box::pin(f(item)));
    }

    while !active.is_empty() {
        let completed = poll_fn(|cx| {
            for idx in 0..active.len() {
                if let Poll::Ready(val) = active[idx].as_mut().poll(cx) {
                    drop(active.swap_remove(idx));
                    return Poll::Ready(val);
                }
            }
            Poll::Pending
        })
        .await;
        results.push(completed);
        if let Some(next_item) = iter.next() {
            active.push(Box::pin(f(next_item)));
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use uuid::Uuid;

    use super::*;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("claude-acp-sdk-{}", Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_jsonl(path: &Path, lines: &[Value]) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut content = String::new();
        for line in lines {
            content.push_str(&serde_json::to_string(line).unwrap());
            content.push('\n');
        }
        std::fs::write(path, content).unwrap();
    }

    #[tokio::test]
    async fn lists_sessions_sorted_with_cjk_and_collision_filtering() {
        let tmp = TempDir::new();
        let store = TranscriptStore::new(&tmp.path);

        let cwd_dot = Path::new("/work/app.v1");
        let cwd_slash = Path::new("/work/app/v1");
        assert_eq!(encode_project_dir(cwd_dot), encode_project_dir(cwd_slash));

        let cjk_title = "重构会话管理 / セッション管理のリファクタリング / 세션 관리 리팩터링";
        let s_older = "11111111-1111-4111-8111-111111111111";
        let s_newer = "22222222-2222-4222-8222-222222222222";
        let s_other = "33333333-3333-4333-8333-333333333333";

        write_jsonl(
            &transcript_path(&tmp.path, cwd_dot, s_older),
            &[json!({
                "type": "user",
                "sessionId": s_older,
                "cwd": cwd_dot,
                "timestamp": "2026-11-17T15:00:05.250Z",
                "message": { "role": "user", "content": [{"type": "text", "text": "Fix framing bug"}] }
            })],
        );
        write_jsonl(
            &transcript_path(&tmp.path, cwd_dot, s_newer),
            &[
                json!({
                    "type": "user",
                    "sessionId": s_newer,
                    "cwd": cwd_dot,
                    "timestamp": "2026-11-17T16:10:00Z",
                    "message": { "role": "user", "content": "Initial prompt" }
                }),
                json!({
                    "type": "system",
                    "subtype": "status",
                    "sessionId": s_newer,
                    "timestamp": "2026-02-31T23:59:59Z"
                }),
                json!({ "type": "ai-title", "sessionId": s_newer, "aiTitle": cjk_title }),
            ],
        );
        write_jsonl(
            &transcript_path(&tmp.path, cwd_slash, s_other),
            &[json!({
                "type": "user",
                "sessionId": s_other,
                "cwd": cwd_slash,
                "timestamp": "2026-11-17T17:00:00.000Z",
                "message": { "role": "user", "content": "Different workspace" }
            })],
        );

        let global = store.list_sessions(None).await.unwrap();
        assert_eq!(global.len(), 3);
        assert_eq!(global[0].session_id, s_other);
        assert_eq!(global[1].session_id, s_newer);
        assert_eq!(global[1].title.as_deref(), Some(cjk_title));
        assert_eq!(
            global[1].updated_at.as_deref(),
            Some("2026-11-17T16:10:00Z")
        );
        assert_eq!(global[2].session_id, s_older);
        assert_eq!(global[2].title.as_deref(), Some("Fix framing bug"));

        let filtered = store.list_sessions(Some(cwd_dot)).await.unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].session_id, s_newer);
        assert_eq!(filtered[1].session_id, s_older);
    }

    #[tokio::test]
    async fn loads_replays_deletes_and_rejects_traversal() {
        let tmp = TempDir::new();
        let store = TranscriptStore::new(&tmp.path);
        let cwd = Path::new("/work/project");
        let sid = "44444444-4444-4444-8444-444444444444";

        write_jsonl(
            &transcript_path(&tmp.path, cwd, sid),
            &[
                json!({
                    "type": "user",
                    "sessionId": sid,
                    "cwd": cwd,
                    "uuid": "u-1",
                    "timestamp": "2026-11-17T16:00:00Z",
                    "message": { "role": "user", "content": "Run check" }
                }),
                json!({
                    "type": "assistant",
                    "sessionId": sid,
                    "cwd": cwd,
                    "uuid": "a-1",
                    "parentUuid": "u-1",
                    "message": {
                        "id": "msg_1",
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [{"type": "text", "text": "All clean."}]
                    }
                }),
                json!({ "type": "atis-latch", "sessionId": sid, "atis": true }),
            ],
        );

        let records = store.load_transcript(sid, Some(cwd)).await.unwrap();
        assert_eq!(records.len(), 3);
        assert!(matches!(
            &records[0].record,
            TranscriptRecord::Output(ClaudeOutput::User(_))
        ));
        assert!(matches!(
            &records[1].record,
            TranscriptRecord::Output(ClaudeOutput::Assistant(_))
        ));
        assert!(matches!(
            &records[2].record,
            TranscriptRecord::Unknown { record_type, .. } if record_type == "atis-latch"
        ));

        assert!(store.delete_transcript(sid, None).await.unwrap());
        assert!(!store.delete_transcript(sid, None).await.unwrap());
        assert!(matches!(
            store.load_transcript("../outside", None).await,
            Err(SdkError::Protocol(_))
        ));
    }
}
