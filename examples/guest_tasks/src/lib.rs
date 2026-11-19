//! Example CPU tasks for struum.
//!
//! Build: `cargo build --release --target wasm32-unknown-unknown`
//! (from this directory, or `scripts/build-guest-tasks.sh`).
//!
//! Buffer indices follow the order the requester adds buffers.

use bytemuck::{Pod, Zeroable};
use struum_guest::TaskContext;

/// Must match the host's `#[gpu_type] struct Param` (repr(C), two f32).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Param {
    pub m: f32,
    pub c: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Gradient {
    pub dm: f32,
    pub dc: f32,
}

/// Linear-regression gradient per sample.
/// Buffers: 0 = param (broadcast, 1 Param), 1 = x, 2 = y, 3 = gradients (output).
fn gradient(ctx: &TaskContext) {
    let p = ctx.input::<Param>(0)[0];
    let x = ctx.input::<f32>(1);
    let y = ctx.input::<f32>(2);
    let g = ctx.output::<Gradient>(3);
    for i in 0..ctx.len() {
        let error = p.m * x[i] + p.c - y[i];
        g[i] = Gradient { dm: 2.0 * x[i] * error, dc: 2.0 * error };
    }
}

/// CPU-heavy demo: counts primes below each limit (trial division).
/// Buffers: 0 = limits (u32), 1 = counts (u32, output).
fn count_primes(ctx: &TaskContext) {
    let limits = ctx.input::<u32>(0);
    let counts = ctx.output::<u32>(1);
    for i in 0..ctx.len() {
        counts[i] = primes_below(limits[i]);
    }
}

/// Shared with the host example so it can verify results.
pub fn primes_below(limit: u32) -> u32 {
    (2..limit).filter(|&n| is_prime(n)).count() as u32
}

fn is_prime(n: u32) -> bool {
    if n < 4 {
        return n >= 2;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

struum_guest::tasks!(gradient, count_primes);
