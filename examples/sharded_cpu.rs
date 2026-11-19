//! Requester that shards CPU (WebAssembly) tasks across every worker it finds.
//!
//! 1. `scripts/build-guest-tasks.sh`  (builds examples/guest_tasks to .wasm)
//! 2. `cargo run -p struum_relay`
//! 3. start 2+ workers: `cargo run -p struum_worker` (use distinct NODE_ID / ports),
//!    and/or the Android worker
//! 4. `cargo run --example sharded_cpu`
//!
//! Env: RELAY_ADDR, MIN_PEERS (2), STRUUM_WASM, PRIME_ITEMS (4000), NODE_UDP_PORT, NODE_TCP_PORT

use std::net::SocketAddr;
use std::time::Instant;

use struum_cpu::CpuTask;
use struum_macros::gpu_type;
use struum_node::{Node, ShardOptions, ShardReport};
use struum_types::{Buffer, StruumError, network::*};

#[gpu_type]
#[derive(Debug)]
struct Param {
    m: f32,
    c: f32,
}

#[gpu_type]
#[derive(Debug)]
struct Gradient {
    dm: f32,
    dc: f32,
}

const DEFAULT_WASM: &str = "examples/guest_tasks/target/wasm32-unknown-unknown/release/struum_guest_tasks.wasm";

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Same algorithm as the guest's `count_primes`, to verify results locally.
fn primes_below(limit: u32) -> u32 {
    let is_prime = |n: u32| {
        if n < 4 {
            return n >= 2;
        }
        if n.is_multiple_of(2) {
            return false;
        }
        let mut d = 3;
        while d * d <= n {
            if n.is_multiple_of(d) {
                return false;
            }
            d += 2;
        }
        true
    };
    (2..limit).filter(|&n| is_prime(n)).count() as u32
}

fn print_report(report: &ShardReport) {
    println!(
        "  {} shard(s) in {:?}, {} reassignment(s), {} peer(s) retired",
        report.shards,
        report.elapsed,
        report.retries,
        report.retired.len()
    );
    let mut peers: Vec<_> = report.completed_by.iter().collect();
    peers.sort();
    for (addr, n) in peers {
        println!("    {addr}: {n} shard(s)");
    }
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
        .parse()
        .expect("Invalid RELAY_ADDR");
    let min_peers: usize = env_or("MIN_PEERS", 2);
    let wasm_path = std::env::var("STRUUM_WASM").unwrap_or_else(|_| DEFAULT_WASM.to_string());
    let wasm = std::fs::read(&wasm_path).map_err(|e| {
        StruumError::NotFound(format!(
            "Cannot read {wasm_path}: {e}. Run scripts/build-guest-tasks.sh first."
        ))
    })?;

    let mut node = Node::<2048>::with_id(
        UID::new(3),
        env_or("NODE_UDP_PORT", 34257),
        env_or("NODE_TCP_PORT", 34267),
        NodeBackend::Cpu,
        1,
    )
    .await?;

    let peers = node
        .discover_peers(relay_addr, min_peers, std::time::Duration::from_secs(2))
        .await?;
    let connected = node.connect_to_peers(&peers).await;
    println!("Connected to {} worker(s): {:?}", connected.len(), connected);

    let opts = ShardOptions::default();

    // --- 1. CPU-heavy job: count primes below many limits ---
    let items: usize = env_or("PRIME_ITEMS", 4000);
    let limits: Vec<u32> = (0..items as u32).map(|i| 20_000 + (i * 7919) % 40_000).collect();
    let mut task = CpuTask::new(wasm.clone(), "count_primes");
    task.add_buffer(Buffer::new("limits", &limits));
    task.add_buffer(Buffer::empty::<u32>("counts", items));
    task.set_work_buffer("limits");

    println!("\nCounting primes for {items} limits across the network...");
    let (work, report) = node.run_sharded_with_report(task, &opts).await?;
    print_report(&report);

    let local_start = Instant::now();
    let expected: Vec<u32> = limits.iter().map(|&l| primes_below(l)).collect();
    let local = local_start.elapsed();
    let out = work.into_cpu_task().expect("CPU result");
    let counts = out.get_buffer("counts").expect("counts").as_slice::<u32>();
    assert_eq!(counts, expected.as_slice(), "network result differs from local");
    println!(
        "  verified against local single-thread run ({:?}): {:.1}x faster",
        local,
        local.as_secs_f64() / report.elapsed.as_secs_f64()
    );

    // --- 2. Iterative job: linear regression, gradients sharded each step ---
    let n = 1000;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let mut params = Param { m: 0.0, c: 0.0 };

    println!("\nLinear regression (y = 2x + 1), {n} samples, gradients sharded per step...");
    let started = Instant::now();
    for iteration in 0..300 {
        let mut task = CpuTask::new(wasm.clone(), "gradient");
        task.add_buffer(Buffer::new("param", &[params]));
        task.add_buffer(Buffer::new("x", &x));
        task.add_buffer(Buffer::new("y", &y));
        task.add_buffer(Buffer::empty::<Gradient>("gradients", n));
        task.set_work_buffer("x");

        let out = node.run_sharded(task, &opts).await?.into_cpu_task().expect("CPU result");
        let grads = out.get_buffer("gradients").expect("gradients").as_slice::<Gradient>();
        let (dm, dc) = grads.iter().fold((0.0, 0.0), |(a, b), g| (a + g.dm, b + g.dc));
        params.m -= 0.5 * dm / n as f32;
        params.c -= 0.5 * dc / n as f32;

        if iteration % 50 == 0 {
            println!("  iteration {iteration}: m={:.4}, c={:.4}", params.m, params.c);
        }
    }
    println!(
        "  final: m = {:.4}, c = {:.4} ({:?} for 300 sharded steps)",
        params.m,
        params.c,
        started.elapsed()
    );

    Ok(())
}
