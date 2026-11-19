//! Requester that shards a GPU kernel across every OpenGL worker it finds.
//!
//! `cargo run -p struum_relay`, start 2+ `struum_worker`s (desktop and/or Android),
//! then `cargo run --example sharded_gpu`.
//!
//! Env: RELAY_ADDR, MIN_PEERS (2), NODE_UDP_PORT, NODE_TCP_PORT

use std::net::SocketAddr;
use std::time::Instant;

use struum_kernel::Kernel;
use struum_macros::gpu_type;
use struum_node::{Node, ShardOptions};
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

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn gradient_kernel(params: &Param, x: &[f32], y: &[f32]) -> Kernel {
    // Shards see their own slice of x/y/gradients; `param` (1 element) is broadcast.
    let source = r#"
        void compute(uint i) {
            Param p = param[0];
            float error = p.m * x[i] + p.c - y[i];
            gradients[i] = Gradient(2.0 * x[i] * error, 2.0 * error);
        }
    "#;
    let mut kernel = Kernel::new(source, "compute");
    kernel.add_buffer::<Param>(Buffer::new("param", &[Param { m: params.m, c: params.c }]));
    kernel.add_buffer::<f32>(Buffer::new("x", x));
    kernel.add_buffer::<f32>(Buffer::new("y", y));
    kernel.add_buffer::<Gradient>(Buffer::empty::<Gradient>("gradients", x.len()));
    kernel.set_work_buffer("x");
    kernel.pack().expect("Failed to pack kernel");
    kernel
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
        .parse()
        .expect("Invalid RELAY_ADDR");
    let min_peers: usize = env_or("MIN_PEERS", 2);

    let mut node = Node::<2048>::with_id(
        UID::new(4),
        env_or("NODE_UDP_PORT", 34258),
        env_or("NODE_TCP_PORT", 34268),
        NodeBackend::Cpu,
        1,
    )
    .await?;

    let peers = node
        .discover_peers(relay_addr, min_peers, std::time::Duration::from_secs(2))
        .await?;
    let connected = node.connect_to_peers(&peers).await;
    let gpu_peers = peers.iter().filter(|p| p.backend == Some(NodeBackend::OpenGL)).count();
    println!("Connected to {} worker(s), {} with OpenGL", connected.len(), gpu_peers);

    let n = 1000;
    let x: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
    let mut params = Param { m: 0.0, c: 0.0 };
    let opts = ShardOptions::default();

    println!("Linear regression (y = 2x + 1), {n} samples, GPU kernel sharded per step...");
    let started = Instant::now();
    for iteration in 0..300 {
        let kernel = gradient_kernel(&params, &x, &y);
        let (work, report) = node.run_sharded_with_report(kernel, &opts).await?;
        let out = work.into_kernel().expect("GPU result");
        let grads = out.get_buffer("gradients").expect("gradients").as_slice::<Gradient>();
        let (dm, dc) = grads.iter().fold((0.0, 0.0), |(a, b), g| (a + g.dm, b + g.dc));
        params.m -= 0.5 * dm / n as f32;
        params.c -= 0.5 * dc / n as f32;

        if iteration % 50 == 0 {
            println!(
                "  iteration {iteration}: m={:.4}, c={:.4} ({} shards over {} peer(s))",
                params.m,
                params.c,
                report.shards,
                report.completed_by.len()
            );
        }
    }
    println!(
        "Final: m = {:.4}, c = {:.4} ({:?} for 300 sharded steps)",
        params.m,
        params.c,
        started.elapsed()
    );
    Ok(())
}
