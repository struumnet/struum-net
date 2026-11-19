//! Example CPU tasks for struum.
//!
//! Build: `cargo build --release --target wasm32-unknown-unknown`
//! (from this directory, or `scripts/build-guest-tasks.sh`).
//!
//! Buffer indices follow the order the requester adds buffers.

use bytemuck::{Pod, Zeroable};
use struum_guest::TaskContext;

pub mod mlp;

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

/// MLP training step for one slice of a minibatch.
/// Buffers: 0 = images (u8 rows, split), 1 = labels (u32, split, work buffer),
/// 2 = config `[input, hidden, classes]`, 3 = params (broadcast),
/// 4 = grads (SumF32, `param_count + 2`: gradient sums, then loss sum and correct count).
fn mlp_grad(ctx: &TaskContext) {
    let images = ctx.rows::<u8>(0);
    let labels = ctx.input::<u32>(1);
    let d = mlp::Dims::from_config(ctx.input::<u32>(2));
    let params = ctx.input::<f32>(3);
    let grads = ctx.output::<f32>(4);
    let p = d.param_count();
    assert_eq!(params.len(), p, "params length does not match config");
    assert_eq!(grads.len(), p + 2, "grads must have param_count + 2 values");
    assert_eq!(ctx.stride(0), d.input, "image rows must have `input` pixels");

    let mut scratch = mlp::Scratch::new(d);
    let (mut loss, mut correct) = (0.0f32, 0.0f32);
    for (pixels, &label) in images.zip(labels) {
        let (l, ok) = mlp::accumulate_grad(d, params, pixels, label, &mut grads[..p], &mut scratch);
        loss += l;
        correct += ok as u32 as f32;
    }
    grads[p] += loss;
    grads[p + 1] += correct;
}

/// MLP inference. Buffers: 0 = images (u8 rows, split), 1 = config, 2 = params
/// (broadcast), 3 = predictions (u32, split, work buffer).
fn mlp_predict(ctx: &TaskContext) {
    let images = ctx.rows::<u8>(0);
    let d = mlp::Dims::from_config(ctx.input::<u32>(1));
    let params = ctx.input::<f32>(2);
    let preds = ctx.output::<u32>(3);
    let mut scratch = mlp::Scratch::new(d);
    for (pixels, pred) in images.zip(preds.iter_mut()) {
        *pred = mlp::predict(d, params, pixels, &mut scratch);
    }
}

struum_guest::tasks!(gradient, count_primes, mlp_grad, mlp_predict);
