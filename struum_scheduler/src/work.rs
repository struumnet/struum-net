use serde_derive::{Deserialize, Serialize};
use std::ops::Range;

use struum_cpu::CpuTask;
use struum_kernel::Kernel;
use struum_types::network::WorkKind;
use struum_types::{Shardable, StruumError};

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
