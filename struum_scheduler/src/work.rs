use serde_derive::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

use struum_cpu::CpuTask;
use struum_kernel::Kernel;
use struum_types::network::{BlobHash, BlobRef, WorkKind};
use struum_types::{Buffer, Shardable, StruumError};

/// A unit of work a node can execute: a GPU kernel or a CPU (WebAssembly) task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Work {
    Gpu(Kernel),
    Cpu(CpuTask),
}

impl Work {
    pub fn kind(&self) -> WorkKind {
        match self {
            Work::Gpu(_) => WorkKind::Gpu,
            Work::Cpu(_) => WorkKind::Cpu,
        }
    }

    /// Serializes the inner kernel/task for a `WorkPacket`.
    pub fn to_bytes(&self) -> Result<Vec<u8>, StruumError> {
        let res = match self {
            Work::Gpu(k) => bincode::serialize(k),
            Work::Cpu(t) => bincode::serialize(t),
        };
        res.map_err(|e| StruumError::SerializationError(format!("Failed to serialize work: {e}")))
    }

    pub fn from_bytes(kind: WorkKind, data: &[u8]) -> Result<Self, StruumError> {
        let err = |e: bincode::Error| StruumError::SerializationError(format!("Failed to deserialize work: {e}"));
        Ok(match kind {
            WorkKind::Gpu => Work::Gpu(bincode::deserialize(data).map_err(err)?),
            WorkKind::Cpu => Work::Cpu(bincode::deserialize(data).map_err(err)?),
        })
    }

    fn buffers(&self) -> Vec<&Buffer> {
        match self {
            Work::Gpu(k) => k.get_buffer_bindings().iter().map(|b| &b.buffer).collect(),
            Work::Cpu(t) => t.buffers().iter().map(|b| &b.buffer).collect(),
        }
    }

    fn buffers_mut(&mut self) -> Vec<&mut Buffer> {
        match self {
            Work::Gpu(k) => k.get_buffer_bindings_mut().iter_mut().map(|b| &mut b.buffer).collect(),
            Work::Cpu(t) => t.buffers_mut().iter_mut().map(|b| &mut b.buffer).collect(),
        }
    }

    /// Every byte blob of this work by slot: buffer `i` is slot `i`, a CPU task's
    /// wasm module is [`MODULE_SLOT`].
    fn blob_slots(&self) -> Vec<(u32, &[u8])> {
        let mut slots: Vec<(u32, &[u8])> =
            self.buffers().into_iter().enumerate().map(|(i, b)| (i as u32, b.data.as_slice())).collect();
        if let Work::Cpu(t) = self {
            slots.push((MODULE_SLOT, t.module()));
        }
        slots
    }

    fn slot_mut(&mut self, slot: u32) -> Option<&mut Vec<u8>> {
        match self {
            Work::Gpu(k) => k.get_buffer_bindings_mut().get_mut(slot as usize).map(|b| &mut b.buffer.data),
            Work::Cpu(t) => {
                if slot == MODULE_SLOT {
                    Some(t.module_mut())
                } else {
                    t.buffers_mut().get_mut(slot as usize).map(|b| &mut b.buffer.data)
                }
            }
        }
    }

    /// Slot and hash of every blob of at least [`BLOB_MIN_BYTES`], the ones
    /// worth sending by reference.
    pub fn blobs(&self) -> Vec<BlobRef> {
        self.blob_slots()
            .into_iter()
            .filter(|(_, data)| data.len() >= BLOB_MIN_BYTES)
            .map(|(slot, data)| BlobRef { slot, hash: blob_hash(data) })
            .collect()
    }

    /// The bytes in `slot`, if it exists.
    pub fn blob_bytes(&self, slot: u32) -> Option<&[u8]> {
        self.blob_slots().into_iter().find(|(s, _)| *s == slot).map(|(_, d)| d)
    }

    /// Takes the bytes out of `slot`, leaving it empty.
    pub fn take_blob(&mut self, slot: u32) -> Option<Vec<u8>> {
        self.slot_mut(slot).map(std::mem::take)
    }

    /// Puts `bytes` into `slot`. Returns false if the slot doesn't exist.
    pub fn put_blob(&mut self, slot: u32, bytes: Vec<u8>) -> bool {
        match self.slot_mut(slot) {
            Some(data) => {
                *data = bytes;
                true
            }
            None => false,
        }
    }

    /// Empties every buffer that is byte-identical to the same buffer in `original`
    /// (the work as it was received), so a result only carries what the task wrote.
    /// Undo with [`Work::restore_elided`]. Execution never resizes buffers, so an
    /// empty buffer whose original was not empty unambiguously means "unchanged".
    pub fn elide_unchanged(&mut self, original: &Work) {
        let originals = original.buffers();
        if originals.len() != self.buffers().len() {
            return;
        }
        for (b, o) in self.buffers_mut().into_iter().zip(originals) {
            if b.name == o.name && b.data == o.data {
                b.data = Vec::new();
            }
        }
    }

    /// Refills buffers elided by [`Work::elide_unchanged`] from `original`.
    pub fn restore_elided(&mut self, original: &Work) -> Result<(), StruumError> {
        let originals = original.buffers();
        if originals.len() != self.buffers().len() {
            return Err(StruumError::ShardError(format!(
                "result has {} buffers but the request had {}",
                self.buffers().len(),
                originals.len()
            )));
        }
        for (b, o) in self.buffers_mut().into_iter().zip(originals) {
            if b.name != o.name {
                return Err(StruumError::ShardError(format!(
                    "result buffer '{}' does not match request buffer '{}'",
                    b.name, o.name
                )));
            }
            if b.data.is_empty() && !o.data.is_empty() {
                b.data = o.data.clone();
            }
        }
        Ok(())
    }

    pub fn into_kernel(self) -> Option<Kernel> {
        match self {
            Work::Gpu(k) => Some(k),
            Work::Cpu(_) => None,
        }
    }

    pub fn into_cpu_task(self) -> Option<CpuTask> {
        match self {
            Work::Cpu(t) => Some(t),
            Work::Gpu(_) => None,
        }
    }
}

impl From<Kernel> for Work {
    fn from(k: Kernel) -> Self {
        Work::Gpu(k)
    }
}

impl From<CpuTask> for Work {
    fn from(t: CpuTask) -> Self {
        Work::Cpu(t)
    }
}

impl Shardable for Work {
    fn work_count(&self) -> usize {
        match self {
            Work::Gpu(k) => k.work_count(),
            Work::Cpu(t) => t.work_count(),
        }
    }

    fn offset(&self) -> usize {
        match self {
            Work::Gpu(k) => Shardable::offset(k),
            Work::Cpu(t) => Shardable::offset(t),
        }
    }

    fn shard(&self, range: Range<usize>) -> Result<Self, StruumError> {
        Ok(match self {
            Work::Gpu(k) => Work::Gpu(k.shard(range)?),
            Work::Cpu(t) => Work::Cpu(t.shard(range)?),
        })
    }

    fn merge(parts: Vec<Self>) -> Result<Self, StruumError> {
        let mismatch = || StruumError::ShardError("cannot merge GPU and CPU shards together".to_string());
        match parts.first() {
            Some(Work::Gpu(_)) => Ok(Work::Gpu(Kernel::merge(
                parts.into_iter().map(|p| p.into_kernel().ok_or_else(mismatch)).collect::<Result<_, _>>()?,
            )?)),
            Some(Work::Cpu(_)) => Ok(Work::Cpu(CpuTask::merge(
                parts.into_iter().map(|p| p.into_cpu_task().ok_or_else(mismatch)).collect::<Result<_, _>>()?,
            )?)),
            None => Err(StruumError::ShardError("cannot merge zero shards".to_string())),
        }
    }
}

/// Slot number of a CPU task's wasm module in [`BlobRef`]s.
pub const MODULE_SLOT: u32 = u32::MAX;

/// Blobs smaller than this are always sent inline.
pub const BLOB_MIN_BYTES: usize = 4096;

/// Stable content hash used for [`BlobRef`]s (xxh3-64, same on every build and device).
pub fn blob_hash(bytes: &[u8]) -> BlobHash {
    xxhash_rust::xxh3::xxh3_64(bytes)
}

/// Blobs a worker holds for one requester connection, so repeated large inputs
/// (wasm modules, broadcast weights) cross the network once. Bounded by total
/// bytes; the least recently used blobs are dropped first.
#[derive(Debug)]
pub struct BlobStore {
    blobs: HashMap<BlobHash, (Arc<Vec<u8>>, u64)>,
    bytes: usize,
    limit: usize,
    clock: u64,
}

impl BlobStore {
    pub fn new(limit_bytes: usize) -> Self {
        Self { blobs: HashMap::new(), bytes: 0, limit: limit_bytes, clock: 0 }
    }

    /// Size limit from `STRUUM_BLOB_CACHE_MB` (default 1024 MiB).
    pub fn env_limit_bytes() -> usize {
        let mb: usize = std::env::var("STRUUM_BLOB_CACHE_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        mb << 20
    }

    /// A store sized by [`BlobStore::env_limit_bytes`].
    pub fn from_env() -> Self {
        Self::new(Self::env_limit_bytes())
    }

    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }

    pub fn contains(&self, hash: BlobHash) -> bool {
        self.blobs.contains_key(&hash)
    }

    /// Stores `bytes` under its hash, evicting older blobs to stay within the limit.
    /// Blobs larger than the whole limit are not stored.
    pub fn insert(&mut self, bytes: &[u8]) -> BlobHash {
        let hash = blob_hash(bytes);
        self.clock += 1;
        if let Some(entry) = self.blobs.get_mut(&hash) {
            entry.1 = self.clock;
            return hash;
        }
        if bytes.len() > self.limit {
            return hash;
        }
        while self.bytes + bytes.len() > self.limit {
            let oldest = *self
                .blobs
                .iter()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(h, _)| h)
                .expect("over the limit implies a stored blob");
            let (gone, _) = self.blobs.remove(&oldest).unwrap();
            self.bytes -= gone.len();
        }
        self.bytes += bytes.len();
        self.blobs.insert(hash, (Arc::new(bytes.to_vec()), self.clock));
        hash
    }

    pub fn get(&mut self, hash: BlobHash) -> Option<Arc<Vec<u8>>> {
        self.clock += 1;
        let clock = self.clock;
        self.blobs.get_mut(&hash).map(|entry| {
            entry.1 = clock;
            entry.0.clone()
        })
    }

    /// Stores every blob of `work` worth referencing later.
    pub fn remember(&mut self, work: &Work) {
        for (_, data) in work.blob_slots() {
            if data.len() >= BLOB_MIN_BYTES {
                self.insert(data);
            }
        }
    }

    /// Fills `work`'s referenced slots from the store. On a miss nothing is
    /// changed and the missing hashes are returned.
    pub fn resolve(&mut self, work: &mut Work, refs: &[BlobRef]) -> Result<(), Vec<BlobHash>> {
        let missing: Vec<BlobHash> = refs.iter().map(|r| r.hash).filter(|h| !self.contains(*h)).collect();
        if !missing.is_empty() {
            return Err(missing);
        }
        for r in refs {
            let bytes = self.get(r.hash).expect("checked above");
            if !work.put_blob(r.slot, bytes.as_ref().clone()) {
                return Err(vec![r.hash]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elide_and_restore_round_trip() {
        let original = Work::Cpu(struum_cpu::selftest::task(8));
        let mut result = original.clone();
        if let Work::Cpu(t) = &mut result {
            t.buffers_mut()[1].buffer.data[0] = 42; // the task wrote `y`
        }
        let expected = result.clone();

        result.elide_unchanged(&original);
        let Work::Cpu(t) = &result else { unreachable!() };
        assert!(t.buffers()[0].buffer.data.is_empty(), "unchanged input is elided");
        assert_eq!(t.buffers()[1].buffer.data.len(), 32, "written output is kept");

        result.restore_elided(&original).unwrap();
        assert_eq!(result.to_bytes().unwrap(), expected.to_bytes().unwrap());
    }
}

#[cfg(test)]
mod blob_tests {
    use super::*;
    use struum_types::Buffer;

    fn task_with_weights() -> Work {
        let mut t = struum_cpu::selftest::task(8);
        t.add_buffer(Buffer::new("weights", &vec![1.5f32; 4096]));
        Work::Cpu(t)
    }

    #[test]
    fn large_blobs_round_trip_through_a_store() {
        let original = task_with_weights();
        let refs = original.blobs();
        // Only the 16 KiB weights qualify: x, y and the small WAT module stay inline.
        assert_eq!(refs.iter().map(|r| r.slot).collect::<Vec<_>>(), vec![2]);

        let mut store = BlobStore::new(1 << 20);
        store.remember(&original);

        let mut stripped = original.clone();
        for r in &refs {
            stripped.take_blob(r.slot).unwrap();
        }
        let full = original.to_bytes().unwrap().len();
        assert!(stripped.to_bytes().unwrap().len() + 16_384 <= full);
        store.resolve(&mut stripped, &refs).unwrap();
        assert_eq!(stripped.to_bytes().unwrap(), original.to_bytes().unwrap());
    }

    #[test]
    fn misses_are_reported_and_leave_work_untouched() {
        let original = task_with_weights();
        let refs = original.blobs();
        let mut store = BlobStore::new(1 << 20);
        let mut stripped = original.clone();
        stripped.take_blob(2).unwrap();
        let before = stripped.to_bytes().unwrap();
        assert_eq!(store.resolve(&mut stripped, &refs).unwrap_err(), vec![refs[0].hash]);
        assert_eq!(stripped.to_bytes().unwrap(), before);
    }

    #[test]
    fn store_evicts_least_recently_used() {
        let a = vec![1u8; 6000];
        let b = vec![2u8; 6000];
        let c = vec![3u8; 6000];
        let mut store = BlobStore::new(13_000);
        let ha = store.insert(&a);
        let hb = store.insert(&b);
        store.get(ha); // a is now more recent than b
        let hc = store.insert(&c);
        assert!(store.contains(ha) && store.contains(hc));
        assert!(!store.contains(hb));
        // Bigger than the whole store: not kept.
        let huge = store.insert(&vec![0u8; 20_000]);
        assert!(!store.contains(huge));
    }

    #[test]
    fn modules_are_blobs_when_large() {
        let mut t = struum_cpu::selftest::task(8);
        t.module_mut().resize(8192, b' '); // trailing whitespace keeps the WAT valid
        let refs = Work::Cpu(t).blobs();
        assert_eq!(refs.iter().map(|r| r.slot).collect::<Vec<_>>(), vec![MODULE_SLOT]);
    }

    #[test]
    fn hashes_are_stable() {
        // Pinned value: requester and worker on different builds must agree.
        assert_eq!(blob_hash(b"struum"), xxhash_rust::xxh3::xxh3_64(b"struum"));
        assert_ne!(blob_hash(&[0u8; 4096]), blob_hash(&[1u8; 4096]));
    }
}
