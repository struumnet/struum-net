use serde_derive::{Deserialize, Serialize};
use std::ops::Range;

use struum_types::shard::{merge_buffers, shard_buffers};
use struum_types::{BoundBuffer, Buffer, ShardMode, Shardable, StruumError};

/// A CPU task: a WebAssembly module, the entry point to call, and its buffers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuTask {
    #[serde(with = "serde_bytes")]
    module: Vec<u8>,
    entry: String,
    buffers: Vec<BoundBuffer>,
    work_buffer: Option<String>,
    offset: u32,
}

impl CpuTask {
    /// `module` is a compiled `.wasm` binary (or WAT text) and `entry` the exported function.
    pub fn new(module: impl Into<Vec<u8>>, entry: &str) -> Self {
        Self {
            module: module.into(),
            entry: entry.to_string(),
            buffers: Vec::new(),
            work_buffer: None,
            offset: 0,
        }
    }

    /// Adds a buffer; the guest sees buffers by index in the order they are added.
    pub fn add_buffer(&mut self, buffer: Buffer) {
        self.buffers.push(BoundBuffer::new(buffer));
    }

    /// The buffer whose element count defines the work range (always `Split`).
    pub fn set_work_buffer(&mut self, name: &str) {
        self.work_buffer = Some(name.to_string());
    }

    /// Overrides how buffer `name` is distributed when the task is sharded.
    pub fn set_shard_mode(&mut self, name: &str, mode: ShardMode) -> Result<(), StruumError> {
        let b = self
            .buffers
            .iter_mut()
            .find(|b| b.buffer.name == name)
            .ok_or_else(|| StruumError::NotFound(format!("Buffer '{}' not found", name)))?;
        b.mode = Some(mode);
        Ok(())
    }

    pub fn get_buffer(&self, name: &str) -> Option<&Buffer> {
        self.buffers.iter().find(|b| b.buffer.name == name).map(|b| &b.buffer)
    }

    pub fn module(&self) -> &[u8] {
        &self.module
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    /// All buffers, in the order the guest sees them.
    pub fn buffers(&self) -> &[BoundBuffer] {
        &self.buffers
    }

    pub fn buffers_mut(&mut self) -> &mut [BoundBuffer] {
        &mut self.buffers
    }

    fn work_len(&self) -> Result<usize, StruumError> {
        let name = self
            .work_buffer
            .as_ref()
            .ok_or_else(|| StruumError::NotFound("CpuTask work buffer not set".to_string()))?;
        self.get_buffer(name)
            .map(|b| b.actual_len())
            .ok_or_else(|| StruumError::NotFound(format!("Work buffer '{}' not found", name)))
    }

    /// Buffers with the work buffer pinned to `Split`.
    fn bound_with_work_split(&self) -> Vec<BoundBuffer> {
        let mut buffers = self.buffers.clone();
        if let Some(work) = &self.work_buffer {
            for b in buffers.iter_mut().filter(|b| &b.buffer.name == work) {
                b.mode = Some(ShardMode::Split);
            }
        }
        buffers
    }
}

impl Shardable for CpuTask {
    fn work_count(&self) -> usize {
        self.work_len().unwrap_or(0)
    }

    fn offset(&self) -> usize {
        self.offset as usize
    }

    fn shard(&self, range: Range<usize>) -> Result<Self, StruumError> {
        let work_count = self.work_len()?;
        Ok(Self {
            module: self.module.clone(),
            entry: self.entry.clone(),
            buffers: shard_buffers(&self.bound_with_work_split(), work_count, range.clone())?,
            work_buffer: self.work_buffer.clone(),
            offset: self.offset + range.start as u32,
        })
    }

    fn merge(parts: Vec<Self>) -> Result<Self, StruumError> {
        let first = parts
            .first()
            .ok_or_else(|| StruumError::ShardError("cannot merge zero shards".to_string()))?;
        let mut merged = Self {
            module: first.module.clone(),
            entry: first.entry.clone(),
            buffers: Vec::new(),
            work_buffer: first.work_buffer.clone(),
            offset: first.offset,
        };
        let counts: Vec<usize> = parts.iter().map(|p| p.work_count()).collect();
        let buffers = parts.into_iter().map(|p| p.bound_with_work_split()).collect();
        merged.buffers = merge_buffers(buffers, &counts)?;
        Ok(merged)
    }
}
