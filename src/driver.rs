//! Executor-independent cooperative polling of per-session SDK I/O drivers.
//!
//! `claude-acp-sdk`'s [`Client`](claude_acp_sdk::Client) separates its handle
//! from the background byte-stream/child-process driver future so no async
//! runtime is assumed. [`DriverSlot`] lets any in-flight `AgentHandler` future
//! advance its session's driver inline without `tokio::spawn`, while still
//! supporting concurrent operations such as mid-turn `session/cancel`
//! (`client.interrupt()`) alongside an active `session/prompt`.

use std::fmt;
use std::future::{Future, poll_fn};
use std::pin::{Pin, pin};
use std::task::Poll;

use async_lock::Mutex;

type BoxDriverFuture = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Synchronized slot holding a session's active SDK I/O driver future.
pub struct DriverSlot {
    inner: Mutex<Option<BoxDriverFuture>>,
}

impl fmt::Debug for DriverSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DriverSlot").finish_non_exhaustive()
    }
}

impl DriverSlot {
    /// Wrap a boxed SDK driver future in a new [`DriverSlot`].
    pub(crate) const fn new(driver: BoxDriverFuture) -> Self {
        Self {
            inner: Mutex::new(Some(driver)),
        }
    }

    /// Poll `op` to completion while ensuring the session driver is actively
    /// polled whenever no other caller currently holds the driver lock.
    ///
    /// When a second caller (for example `session/cancel` invoking
    /// `client.interrupt()`) enters while `session/prompt` already holds the
    /// driver lock, `op` queues its outbound frame immediately, waking the
    /// active driver holder, and hands off driver ownership seamlessly if the
    /// first caller finishes earlier.
    pub(crate) async fn drive_until<F, T>(&self, op: F) -> T
    where
        F: Future<Output = T>,
    {
        let mut op = pin!(op);
        let mut lock_fut = pin!(self.inner.lock());
        let mut guard_opt = None;

        poll_fn(|cx| {
            loop {
                if let Poll::Ready(output) = op.as_mut().poll(cx) {
                    return Poll::Ready(output);
                }

                if guard_opt.is_none() {
                    match lock_fut.as_mut().poll(cx) {
                        Poll::Ready(guard) => {
                            guard_opt = Some(guard);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }

                let guard = guard_opt.as_mut().expect("driver mutex guard acquired");
                if let Some(driver) = guard.as_mut() {
                    match driver.as_mut().poll(cx) {
                        Poll::Ready(()) => {
                            **guard = None;
                            // Driver just closed its channels; re-poll `op` immediately.
                        }
                        Poll::Pending => {
                            return op.as_mut().poll(cx);
                        }
                    }
                } else {
                    return op.as_mut().poll(cx);
                }
            }
        })
        .await
    }

    /// Drive the underlying SDK session future until it terminates cleanly.
    pub(crate) async fn drive_to_completion(&self) {
        let mut guard = self.inner.lock().await;
        if let Some(mut driver) = guard.take() {
            driver.as_mut().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures_lite::future::zip;

    use super::*;

    #[tokio::test]
    async fn concurrent_callers_share_driver_without_deadlock() {
        let (req_tx, req_rx) = async_channel::unbounded::<async_channel::Sender<u32>>();
        let driver = Box::pin(async move {
            let mut count = 0_u32;
            while let Ok(reply) = req_rx.recv().await {
                count += 1;
                let _ = reply.send(count).await;
                if count == 2 {
                    break;
                }
            }
        });

        let slot = Arc::new(DriverSlot::new(driver));
        let tx1 = req_tx.clone();
        let tx2 = req_tx;

        let slot1 = Arc::clone(&slot);
        let slot2 = Arc::clone(&slot);

        let (r1, r2) = zip(
            slot1.drive_until(async move {
                let (reply_tx, reply_rx) = async_channel::bounded(1);
                tx1.send(reply_tx).await.unwrap();
                reply_rx.recv().await.unwrap()
            }),
            slot2.drive_until(async move {
                let (reply_tx, reply_rx) = async_channel::bounded(1);
                tx2.send(reply_tx).await.unwrap();
                reply_rx.recv().await.unwrap()
            }),
        )
        .await;

        assert_eq!(r1 + r2, 3);
        slot.drive_to_completion().await;
    }
}
