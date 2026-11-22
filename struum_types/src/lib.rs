//! Types shared by every struum crate.
//!
//! - [`Buffer`]: a named byte buffer with an element stride, the data of every job.
//! - [`shard`]: splitting a job's work range into shards and merging the results
//!   ([`ShardMode`], [`BoundBuffer`], [`Shardable`], [`split_ranges`]).
//! - [`network`]: node identity ([`UID`](network::UID),
//!   [`NodeDetails`](network::NodeDetails)) and the UDP/TCP packets exchanged between nodes.
//! - [`GPUType`]: maps a Rust type to its GLSL type (derive it with `#[gpu_type]` from
//!   `struum_macros`).
//! - [`StruumError`]: the error type used throughout.

pub mod buffer;
pub mod gpu_type;
pub mod network;
pub mod shard;
pub mod struum_error;

pub use buffer::Buffer;
pub use gpu_type::{GPUType, is_builtin_gpu_type};
pub use shard::{BoundBuffer, ShardMode, Shardable, split_ranges};
pub use struum_error::StruumError;
