//! CPU tasks: WebAssembly modules shipped over the network with their data,
//! the same way GPU kernels ship GLSL.
//!
//! # Guest ABI
//! A task module exports:
//! - `memory`
//! - `struum_alloc(size: u32, align: u32) -> u32`
//! - one or more entry points `entry(desc: u32) -> i32` (0 = success)
//!
//! `desc` points to little-endian u32s:
//! `[offset, len, buf_count, (ptr, byte_len, stride) * buf_count]`
//! where `offset..offset+len` is the global work range of this piece and the
//! buffers are in the order they were added to the [`CpuTask`]. `Split`
//! buffers hold only this piece's elements (local index `i` == global `offset + i`).
//!
//! The `struum_guest` crate implements this ABI for Rust guests.

mod executor;
pub mod selftest;
mod task;

pub use executor::CpuExecutor;
pub use task::CpuTask;
