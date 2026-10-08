use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::{Error, Result};

type Hook = Box<dyn Fn() + Send + Sync>;

/// Cooperative cancellation shared between the UI thread, the pipeline and running inferences.
///
/// Long loops call [`CancelToken::check`]; ONNX Runtime runs register a terminate hook so a
/// cancel interrupts an in-flight inference instead of waiting for it to finish.
#[derive(Clone, Default)]
pub struct CancelToken {
    inner: Arc<Inner>,
}

#[derive(Default)]
struct Inner {
    flag: AtomicBool,
    next: AtomicU64,
    hooks: Mutex<Vec<(u64, Hook)>>,
}

/// Removes its hook when dropped.
pub struct HookGuard {
    token: CancelToken,
    id: u64,
}

impl Drop for HookGuard {
    fn drop(&mut self) {
        self.token.inner.hooks.lock().retain(|(i, _)| *i != self.id);
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        if !self.inner.flag.swap(true, Ordering::SeqCst) {
            for (_, h) in self.inner.hooks.lock().iter() {
                h();
            }
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.flag.load(Ordering::Relaxed)
    }

    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Run `f` when the token is cancelled (immediately if it already is) for as long as the
    /// returned guard lives.
    pub fn on_cancel(&self, f: impl Fn() + Send + Sync + 'static) -> HookGuard {
        let id = self.inner.next.fetch_add(1, Ordering::Relaxed);
        // Register first, then look at the flag: a `cancel()` racing with this call either sees
        // the hook in the list or has already set the flag we read here (never neither).
        let mut hooks = self.inner.hooks.lock();
        hooks.push((id, Box::new(f)));
        if self.inner.flag.load(Ordering::SeqCst) {
            (hooks[hooks.len() - 1].1)();
        }
        drop(hooks);
        HookGuard { token: self.clone(), id }
    }
}

/// Progress callback: fraction 0..=1 of the current step plus a short label.
pub type Progress<'a> = &'a (dyn Fn(f32, &str) + Sync);

pub fn no_progress(_: f32, _: &str) {}
