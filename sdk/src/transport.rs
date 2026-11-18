//! LF-only (`\n`) newline-delimited JSON duplex transport.
//!
//! Claude Code's `stream-json` wire format frames messages strictly on ASCII
//! LF (`0x0A`). JSON string payloads may contain unescaped Unicode line/paragraph
//! separators (`U+2028` and `U+2029`), so framing must never split or trim on
//! Unicode line breaks.
//!
//! [`NdjsonReader::next_line`] buffers partial frames inside the reader itself
//! so dropping an in-flight read future inside a `select`/`or` branch never
//! discards already-read bytes.

use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::task::{Context, Poll};

use claude_codes::ClaudeOutput;
use futures_lite::io::{AsyncBufRead, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, BufWriter};
use serde::Serialize;
use tracing::trace;

use crate::error::{Result, SdkError};

/// Default read buffer size (64 KiB) before per-frame dynamic growth.
const READ_BUF_CAPACITY: usize = 64 * 1024;

/// Default write buffer size (16 KiB) flushed once per outbound NDJSON frame.
const WRITE_BUF_CAPACITY: usize = 16 * 1024;

/// Cancel-safe LF-delimited NDJSON frame reader over any [`AsyncRead`].
pub struct NdjsonReader<R> {
    reader: BufReader<R>,
    buf: Vec<u8>,
}

impl<R> fmt::Debug for NdjsonReader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdjsonReader")
            .field("buffered_bytes", &self.buf.len())
            .finish_non_exhaustive()
    }
}

impl<R: AsyncRead + Unpin + Send> NdjsonReader<R> {
    /// Wrap an [`AsyncRead`] stream in a buffered LF-only NDJSON reader.
    #[must_use]
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(READ_BUF_CAPACITY, reader),
            buf: Vec::new(),
        }
    }

    /// Read the next non-blank LF-delimited UTF-8 line, or `Ok(None)` at EOF.
    ///
    /// Only ASCII LF (`\n`) terminates a frame (with an optional preceding `\r`
    /// stripped for CRLF pipes). Unescaped `U+2028` and `U+2029` characters are
    /// preserved intact. Dropping the returned future before completion retains
    /// any bytes already read inside `self`.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Io`] on underlying read errors or
    /// [`SdkError::Protocol`] if a frame is not valid UTF-8.
    pub async fn next_line(&mut self) -> Result<Option<String>> {
        poll_fn(|cx| self.poll_next_line(cx)).await
    }

    /// Read and decode the next [`ClaudeOutput`] frame from the stream, or
    /// `Ok(None)` at EOF.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Io`] on read failure or [`SdkError::Protocol`] when
    /// the frame cannot be parsed as a valid [`ClaudeOutput`].
    pub async fn recv_output(&mut self) -> Result<Option<ClaudeOutput>> {
        let Some(line) = self.next_line().await? else {
            return Ok(None);
        };
        trace!(bytes = line.len(), "received NDJSON frame");
        let output = ClaudeOutput::parse_json_tolerant(&line)?;
        Ok(Some(output))
    }

    fn poll_next_line(&mut self, cx: &mut Context<'_>) -> Poll<Result<Option<String>>> {
        loop {
            let available = match Pin::new(&mut self.reader).poll_fill_buf(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(err)) => return Poll::Ready(Err(SdkError::from(err))),
                Poll::Ready(Ok(buf)) => buf,
            };

            if available.is_empty() {
                if self.buf.is_empty() {
                    return Poll::Ready(Ok(None));
                }
                let raw = std::mem::take(&mut self.buf);
                return match finish_frame(raw) {
                    Ok(Some(line)) => Poll::Ready(Ok(Some(line))),
                    Ok(None) => Poll::Ready(Ok(None)),
                    Err(err) => Poll::Ready(Err(err)),
                };
            }

            if let Some(newline_pos) = available.iter().position(|&b| b == b'\n') {
                self.buf.extend_from_slice(&available[..newline_pos]);
                Pin::new(&mut self.reader).consume(newline_pos + 1);
                let raw = std::mem::take(&mut self.buf);
                match finish_frame(raw) {
                    Ok(Some(line)) => return Poll::Ready(Ok(Some(line))),
                    Ok(None) => continue,
                    Err(err) => return Poll::Ready(Err(err)),
                }
            }

            let len = available.len();
            self.buf.extend_from_slice(available);
            Pin::new(&mut self.reader).consume(len);
        }
    }
}

fn finish_frame(mut raw: Vec<u8>) -> Result<Option<String>> {
    if raw.last() == Some(&b'\r') {
        raw.pop();
    }
    if raw.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    String::from_utf8(raw)
        .map(Some)
        .map_err(|err| SdkError::Protocol(format!("NDJSON frame is not valid UTF-8: {err}")))
}

/// LF-delimited NDJSON frame writer over any [`AsyncWrite`].
pub struct NdjsonWriter<W> {
    writer: BufWriter<W>,
}

impl<W> fmt::Debug for NdjsonWriter<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdjsonWriter").finish_non_exhaustive()
    }
}

impl<W: AsyncWrite + Unpin + Send> NdjsonWriter<W> {
    /// Wrap an [`AsyncWrite`] sink in a buffered LF-only NDJSON writer.
    #[must_use]
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::with_capacity(WRITE_BUF_CAPACITY, writer),
        }
    }

    /// Write one pre-serialized JSON frame terminated by `\n` and flush.
    ///
    /// Rejects embedded `\r` or `\n` bytes while preserving unescaped `U+2028`
    /// and `U+2029` characters verbatim.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Protocol`] if `line` contains a literal `\r` or `\n`,
    /// or [`SdkError::Io`] if writing or flushing fails.
    pub async fn write_raw_line(&mut self, line: &str) -> Result<()> {
        if line.as_bytes().iter().any(|&b| b == b'\n' || b == b'\r') {
            return Err(SdkError::Protocol(
                "outbound NDJSON frame contains embedded CR or LF".to_string(),
            ));
        }
        trace!(bytes = line.len(), "sending NDJSON frame");
        self.writer.write_all(line.as_bytes()).await?;
        self.writer.write_all(b"\n").await?;
        self.writer.flush().await?;
        Ok(())
    }

    /// Serialize `value` as single-line JSON (preserving unescaped `U+2028` and
    /// `U+2029`), append `\n`, and flush.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Json`] if serialization fails or [`SdkError::Io`] if
    /// writing to the underlying sink fails.
    pub fn send_json<T: Serialize>(
        &mut self,
        value: &T,
    ) -> impl Future<Output = Result<()>> + Send {
        let serialized =
            serde_json::to_string(value).map_err(|err| SdkError::json("send_json", err));
        async move { self.write_raw_line(&serialized?).await }
    }

    /// Flush and close the underlying [`AsyncWrite`] sink.
    ///
    /// # Errors
    ///
    /// Returns [`SdkError::Io`] if flushing or closing fails.
    pub async fn close(&mut self) -> Result<()> {
        self.writer.flush().await?;
        self.writer.close().await?;
        Ok(())
    }
}

/// Paired LF-only NDJSON duplex transport over [`AsyncRead`] and [`AsyncWrite`].
#[derive(Debug)]
pub struct NdjsonTransport<R, W> {
    reader: NdjsonReader<R>,
    writer: NdjsonWriter<W>,
}

impl<R, W> NdjsonTransport<R, W>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    /// Create a new duplex NDJSON transport from separate read and write halves.
    #[must_use]
    pub fn new(read: R, write: W) -> Self {
        Self {
            reader: NdjsonReader::new(read),
            writer: NdjsonWriter::new(write),
        }
    }

    /// Split into independent [`NdjsonReader`] and [`NdjsonWriter`] halves.
    #[must_use]
    pub fn split(self) -> (NdjsonReader<R>, NdjsonWriter<W>) {
        (self.reader, self.writer)
    }

    /// Mutable reference to the inbound [`NdjsonReader`].
    pub const fn reader_mut(&mut self) -> &mut NdjsonReader<R> {
        &mut self.reader
    }

    /// Mutable reference to the outbound [`NdjsonWriter`].
    pub const fn writer_mut(&mut self) -> &mut NdjsonWriter<W> {
        &mut self.writer
    }

    /// Read the next LF-delimited line from the inbound half.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`NdjsonReader::next_line`].
    pub async fn next_line(&mut self) -> Result<Option<String>> {
        self.reader.next_line().await
    }

    /// Read and decode the next [`ClaudeOutput`] frame from the inbound half.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`NdjsonReader::recv_output`].
    pub async fn recv_output(&mut self) -> Result<Option<ClaudeOutput>> {
        self.reader.recv_output().await
    }

    /// Serialize and send `value` on the outbound half.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`NdjsonWriter::send_json`].
    pub fn send_json<T: Serialize>(
        &mut self,
        value: &T,
    ) -> impl Future<Output = Result<()>> + Send {
        self.writer.send_json(value)
    }

    /// Flush and close the outbound half.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`NdjsonWriter::close`].
    pub async fn close(&mut self) -> Result<()> {
        self.writer.close().await
    }
}

#[cfg(test)]
mod tests {
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    use futures_lite::stream::Stream;
    use serde_json::{Value, json};

    use super::*;

    struct ChannelPipe {
        rx: Pin<Box<async_channel::Receiver<Vec<u8>>>>,
        tx: async_channel::Sender<Vec<u8>>,
        read_chunk: Vec<u8>,
        read_pos: usize,
    }

    fn memory_pipe() -> (ChannelPipe, ChannelPipe) {
        let (tx_a, rx_b) = async_channel::unbounded();
        let (tx_b, rx_a) = async_channel::unbounded();
        (
            ChannelPipe {
                rx: Box::pin(rx_a),
                tx: tx_a,
                read_chunk: Vec::new(),
                read_pos: 0,
            },
            ChannelPipe {
                rx: Box::pin(rx_b),
                tx: tx_b,
                read_chunk: Vec::new(),
                read_pos: 0,
            },
        )
    }

    impl AsyncRead for ChannelPipe {
        fn poll_read(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &mut [u8],
        ) -> Poll<std::io::Result<usize>> {
            let this = &mut *self;
            if this.read_pos < this.read_chunk.len() {
                let remaining = &this.read_chunk[this.read_pos..];
                let to_copy = remaining.len().min(buf.len());
                buf[..to_copy].copy_from_slice(&remaining[..to_copy]);
                this.read_pos += to_copy;
                if this.read_pos == this.read_chunk.len() {
                    this.read_chunk.clear();
                    this.read_pos = 0;
                }
                return Poll::Ready(Ok(to_copy));
            }
            match this.rx.as_mut().poll_next(cx) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(None) => Poll::Ready(Ok(0)),
                Poll::Ready(Some(chunk)) => {
                    let to_copy = chunk.len().min(buf.len());
                    buf[..to_copy].copy_from_slice(&chunk[..to_copy]);
                    if to_copy < chunk.len() {
                        this.read_chunk = chunk;
                        this.read_pos = to_copy;
                    }
                    Poll::Ready(Ok(to_copy))
                }
            }
        }
    }

    impl AsyncWrite for ChannelPipe {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            if buf.is_empty() {
                return Poll::Ready(Ok(0));
            }
            self.tx
                .try_send(buf.to_vec())
                .map(|()| buf.len())
                .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe closed"))
                .into()
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }

        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            self.tx.close();
            Poll::Ready(Ok(()))
        }
    }

    #[tokio::test]
    async fn preserves_unescaped_u2028_and_u2029_inside_strings() {
        let (a, mut b) = memory_pipe();
        let mut writer = NdjsonWriter::new(a);
        let mut reader = NdjsonReader::new(&mut b);

        let text = "before\u{2028}middle\u{2029}after";
        let payload = json!({
            "type": "system",
            "subtype": "init",
            "session_id": "s-1",
            "note": text,
        });

        writer.send_json(&payload).await.unwrap();
        let raw_line = reader.next_line().await.unwrap().unwrap();
        assert!(raw_line.contains('\u{2028}'));
        assert!(raw_line.contains('\u{2029}'));

        let decoded: Value = serde_json::from_str(&raw_line).unwrap();
        assert_eq!(decoded["note"].as_str(), Some(text));
    }

    #[tokio::test]
    async fn reader_is_cancel_safe_across_dropped_futures() {
        let (mut writer, reader) = memory_pipe();
        let mut ndjson = NdjsonReader::new(reader);

        // Write only the first half of a JSON frame (no trailing '\n' yet).
        writer.write_all(br#"{"type":"system","#).await.unwrap();

        {
            let mut first_attempt = pin!(ndjson.next_line());
            let mut cx = Context::from_waker(Waker::noop());
            assert!(matches!(
                first_attempt.as_mut().poll(&mut cx),
                Poll::Pending
            ));
            // `first_attempt` is dropped here; buffered prefix must survive.
        }

        writer
            .write_all(b"\"subtype\":\"init\",\"session_id\":\"s-cancel\"}\n")
            .await
            .unwrap();

        let output = ndjson.recv_output().await.unwrap().unwrap();
        assert!(output.is_system_init());
        assert_eq!(output.session_id(), Some("s-cancel"));
    }

    #[tokio::test]
    async fn rejects_embedded_newlines_on_write() {
        let (a, _b) = memory_pipe();
        let mut writer = NdjsonWriter::new(a);
        let err = writer
            .write_raw_line("{\"a\":1}\n{\"b\":2}")
            .await
            .unwrap_err();
        assert!(matches!(err, SdkError::Protocol(_)));
    }
}
