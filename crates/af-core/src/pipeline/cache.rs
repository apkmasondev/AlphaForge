//! In-memory cache for expensive pipeline stages (AI mattes, upscales).

use std::sync::Arc;

use lru::LruCache;
use parking_lot::Mutex;

use crate::ai::matting::Matte;
use crate::Rgba;

#[derive(Clone)]
enum Entry {
    Image(Arc<Rgba>),
    Matte(Arc<Matte>),
}

impl Entry {
    fn bytes(&self) -> usize {
        match self {
            Entry::Image(i) => i.as_raw().len(),
            Entry::Matte(m) => m.alpha.len() * 4,
        }
    }
}

pub struct StageCache {
    inner: Mutex<Inner>,
}

struct Inner {
    lru: LruCache<u64, (u64, Entry)>,
    used: usize,
    budget: usize,
}

impl StageCache {
    /// `budget` in bytes.
    pub fn new(budget: usize) -> Self {
        Self { inner: Mutex::new(Inner { lru: LruCache::unbounded(), used: 0, budget }) }
    }

    pub fn set_budget(&self, budget: usize) {
        let mut g = self.inner.lock();
        g.budget = budget;
        Self::evict(&mut g);
    }

    fn evict(g: &mut Inner) {
        while g.used > g.budget {
            match g.lru.pop_lru() {
                Some((_, (_, e))) => g.used -= e.bytes(),
                None => break,
            }
        }
    }

    fn put(&self, key: u64, item: u64, e: Entry) {
        let mut g = self.inner.lock();
        let b = e.bytes();
        if b > g.budget {
            return;
        }
        if let Some((_, old)) = g.lru.put(key, (item, e)) {
            g.used -= old.bytes();
        }
        g.used += b;
        Self::evict(&mut g);
    }

    pub fn get_image(&self, key: u64) -> Option<Arc<Rgba>> {
        match self.inner.lock().lru.get(&key) {
            Some((_, Entry::Image(i))) => Some(Arc::clone(i)),
            _ => None,
        }
    }

    pub fn put_image(&self, key: u64, item: u64, img: Arc<Rgba>) {
        self.put(key, item, Entry::Image(img));
    }

    pub fn get_matte(&self, key: u64) -> Option<Arc<Matte>> {
        match self.inner.lock().lru.get(&key) {
            Some((_, Entry::Matte(m))) => Some(Arc::clone(m)),
            _ => None,
        }
    }

    pub fn put_matte(&self, key: u64, item: u64, m: Arc<Matte>) {
        self.put(key, item, Entry::Matte(m));
    }

    /// Drop everything that belongs to one source item.
    pub fn invalidate_item(&self, item: u64) {
        let mut g = self.inner.lock();
        let keys: Vec<u64> = g.lru.iter().filter(|(_, (i, _))| *i == item).map(|(k, _)| *k).collect();
        for k in keys {
            if let Some((_, e)) = g.lru.pop(&k) {
                g.used -= e.bytes();
            }
        }
    }

    pub fn clear(&self) {
        let mut g = self.inner.lock();
        g.lru.clear();
        g.used = 0;
    }

    pub fn used_bytes(&self) -> usize {
        self.inner.lock().used
    }
}
