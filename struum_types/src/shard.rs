//! Sharding primitives shared by GPU kernels and CPU tasks.
//!
//! A job covers a work range `0..work_count`. Sharding splits that range into
//! contiguous sub-ranges. Buffers marked [`ShardMode::Split`] are sliced to the
//! sub-range (local index `i` == global index `offset + i`), buffers marked
//! [`ShardMode::Broadcast`] are sent whole to every shard. Merging concatenates
//! the `Split` buffers back in range order.
//!
//! Outputs must be `Split` (map-style work). Reductions are done by the caller
//! after merging.

use serde_derive::{Deserialize, Serialize};
use std::ops::Range;

use crate::{Buffer, StruumError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardMode {
    /// Sliced per shard: one element per work item.
    Split,
    /// Copied whole to every shard (parameters, lookup tables, ...).
    Broadcast,
}

impl ShardMode {
    /// Default rule: a buffer with exactly one element per work item is split.
    pub fn infer(buffer: &Buffer, work_count: usize) -> Self {
        if buffer.actual_len() == work_count {
            ShardMode::Split
        } else {
            ShardMode::Broadcast
        }
    }
}

/// A buffer together with how it is distributed across shards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundBuffer {
    pub buffer: Buffer,
    /// Explicit mode; `None` means "infer from the work count".
    pub mode: Option<ShardMode>,
}

impl BoundBuffer {
    pub fn new(buffer: Buffer) -> Self {
        Self { buffer, mode: None }
    }

    pub fn resolved_mode(&self, work_count: usize) -> ShardMode {
        self.mode
            .unwrap_or_else(|| ShardMode::infer(&self.buffer, work_count))
    }
}

/// Work that can be split into contiguous shards and merged back.
pub trait Shardable: Sized {
    /// Number of work items (the global range is `0..work_count`).
    fn work_count(&self) -> usize;

    /// Global index of this piece's first work item (0 for an unsharded job).
    fn offset(&self) -> usize;

    /// Creates the piece of work covering `range` (relative to this piece).
    fn shard(&self, range: Range<usize>) -> Result<Self, StruumError>;

    /// Merges pieces produced by [`Shardable::shard`], given in range order.
    fn merge(parts: Vec<Self>) -> Result<Self, StruumError>;
}

/// Splits `0..count` into at most `n` balanced, non-empty, contiguous ranges.
pub fn split_ranges(count: usize, n: usize) -> Vec<Range<usize>> {
    let n = n.max(1).min(count);
    if n == 0 {
        return Vec::new();
    }
    let base = count / n;
    let extra = count % n;
    let mut start = 0;
    (0..n)
        .map(|i| {
            let len = base + usize::from(i < extra);
            let range = start..start + len;
            start += len;
            range
        })
        .collect()
}

/// Slices `buffers` for `range`, given the parent's `work_count`.
pub fn shard_buffers(
    buffers: &[BoundBuffer],
    work_count: usize,
    range: Range<usize>,
) -> Result<Vec<BoundBuffer>, StruumError> {
    buffers
        .iter()
        .map(|b| {
            let mode = b.resolved_mode(work_count);
            let buffer = match mode {
                ShardMode::Split => {
                    if b.buffer.actual_len() != work_count {
                        return Err(StruumError::ShardError(format!(
                            "split buffer '{}' has {} elements but the work count is {}",
                            b.buffer.name,
                            b.buffer.actual_len(),
                            work_count
                        )));
                    }
                    b.buffer.slice(range.clone())?
                }
                ShardMode::Broadcast => b.buffer.clone(),
            };
            // Pin the mode so shards of shards don't re-infer with a smaller count.
            Ok(BoundBuffer { buffer, mode: Some(mode) })
        })
        .collect()
}

/// Merges per-shard buffer lists (all in the same binding order).
/// `Split` buffers are concatenated; `Broadcast` buffers are taken from the first shard.
pub fn merge_buffers(
    parts: Vec<Vec<BoundBuffer>>,
    work_counts: &[usize],
) -> Result<Vec<BoundBuffer>, StruumError> {
    let first = parts
        .first()
        .ok_or_else(|| StruumError::ShardError("cannot merge zero shards".to_string()))?;
    if parts.iter().any(|p| p.len() != first.len()) {
        return Err(StruumError::ShardError(
            "shards have different buffer layouts".to_string(),
        ));
    }

    (0..first.len())
        .map(|i| {
            let mode = parts[0][i].resolved_mode(work_counts[0]);
            let buffer = match mode {
                ShardMode::Split => {
                    let pieces: Vec<Buffer> = parts.iter().map(|p| p[i].buffer.clone()).collect();
                    Buffer::concat(&pieces)?
                }
                ShardMode::Broadcast => parts[0][i].buffer.clone(),
            };
            Ok(BoundBuffer { buffer, mode: parts[0][i].mode })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_ranges_balanced() {
        assert_eq!(split_ranges(10, 3), vec![0..4, 4..7, 7..10]);
        assert_eq!(split_ranges(2, 5), vec![0..1, 1..2]);
        assert_eq!(split_ranges(0, 4), Vec::<Range<usize>>::new());
        assert_eq!(split_ranges(7, 0), vec![0..7]);
    }

    #[test]
    fn slice_and_concat_round_trip() {
        let data: Vec<u32> = (0..10).collect();
        let buf = Buffer::new("x", &data);
        let parts: Vec<Buffer> = split_ranges(10, 3)
            .into_iter()
            .map(|r| buf.slice(r).unwrap())
            .collect();
        assert_eq!(parts[1].as_slice::<u32>(), &[4, 5, 6]);
        let joined = Buffer::concat(&parts).unwrap();
        assert_eq!(joined.as_slice::<u32>(), data.as_slice());
        assert!(buf.slice(8..11).is_err());
    }

    #[test]
    fn shard_and_merge_buffers() {
        let x: Vec<f32> = (0..6).map(|v| v as f32).collect();
        let bufs = vec![
            BoundBuffer::new(Buffer::new("x", &x)),
            BoundBuffer::new(Buffer::new("param", &[1.0f32, 2.0])),
        ];
        let shards: Vec<_> = split_ranges(6, 2)
            .into_iter()
            .map(|r| shard_buffers(&bufs, 6, r).unwrap())
            .collect();
        assert_eq!(shards[1][0].buffer.as_slice::<f32>(), &[3.0, 4.0, 5.0]);
        assert_eq!(shards[1][1].buffer.as_slice::<f32>(), &[1.0, 2.0]);
        let merged = merge_buffers(shards, &[3, 3]).unwrap();
        assert_eq!(merged[0].buffer.as_slice::<f32>(), x.as_slice());
        assert_eq!(merged[1].buffer.as_slice::<f32>(), &[1.0, 2.0]);
    }
}
