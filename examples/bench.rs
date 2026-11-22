//! Benchmark for sharded jobs: wall time and bytes on the wire.
//!
//! By default it starts `PEERS` CPU workers in this process (localhost TCP), so it needs
//! no relay, devices or wasm build. Over localhost, bandwidth is effectively free, so the
//! numbers to compare across versions are the MB moved; wall time matters on a real LAN
//! (set `RELAY_ADDR` to use discovered workers instead).
//!
//! Scenarios (all use the built-in selftest module, `y[i] = 2 * x[i] + offset + i`):
//! - `split`:     split input and output only, so transfer is proportional to the data.
//! - `broadcast`: adds a `WEIGHTS_MB` buffer sent whole with every shard, run for `STEPS`
//!                iterations, the traffic pattern of data-parallel training (MNIST).
//!
//! Env: SCENARIO (all|split|broadcast), PEERS (4), ITEMS (1_000_000), WEIGHTS_MB (8),
//!      STEPS (5), SHARDS_PER_PEER (2), BLOB_CACHE (1; 0 sends every blob in full),
//!      IN_FLIGHT (2; shards outstanding per peer), SPECULATE (1; 0 turns off re-running stragglers),
//!      BENCH_JSON (path), RELAY_ADDR, MIN_PEERS
//!
//! `cargo run --release --example bench`

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use struum_cpu::{CpuTask, selftest};
use struum_node::{Node, ShardOptions, ShardReport};
use struum_types::network::{NodeBackend, NodeDetails, UID};
use struum_types::{Buffer, StruumError};

type BenchNode = Node<2048>;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

async fn spawn_local_worker() -> Result<NodeDetails, StruumError> {
    let mut node = BenchNode::with_id(UID::generate(), 0, 0, NodeBackend::Cpu, 1).await?;
    let mut details = node.details();
    details.ip = SocketAddr::new("127.0.0.1".parse().unwrap(), details.ip.port());
    tokio::spawn(async move {
        loop {
            let Ok(peer) = node.accept_peer_connection().await else { return };
            if node.serve_peer(peer).await.is_err() {
                return;
            }
        }
    });
    Ok(details)
}

struct Row {
    scenario: &'static str,
    peers: usize,
    steps: usize,
    wall: Duration,
    sent: u64,
    received: u64,
    saved: u64,
    shards: usize,
    retries: usize,
}

impl Row {
    fn print(&self) {
        println!(
            "{:<10} {:>5} {:>5} {:>10.1} {:>10.2} {:>10.2} {:>10.2} {:>7} {:>7}",
            self.scenario,
            self.peers,
            self.steps,
            self.wall.as_secs_f64() * 1e3,
            self.sent as f64 / 1e6,
            self.received as f64 / 1e6,
            self.saved as f64 / 1e6,
            self.shards,
            self.retries
        );
    }

    fn json(&self) -> String {
        format!(
            r#"{{"scenario":"{}","peers":{},"steps":{},"wall_ms":{:.3},"mb_sent":{:.4},"mb_received":{:.4},"mb_saved":{:.4},"shards":{},"retries":{}}}"#,
            self.scenario,
            self.peers,
            self.steps,
            self.wall.as_secs_f64() * 1e3,
            self.sent as f64 / 1e6,
            self.received as f64 / 1e6,
            self.saved as f64 / 1e6,
            self.shards,
            self.retries
        )
    }
}

fn task(items: usize, weights_mb: usize) -> CpuTask {
    let mut task = selftest::task(items);
    if weights_mb > 0 {
        // Not read by the module: it only stands in for parameters every shard needs.
        let weights: Vec<f32> = (0..weights_mb * 262_144).map(|i| i as f32).collect();
        task.add_buffer(Buffer::new("weights", &weights));
    }
    task
}

async fn run(
    node: &mut BenchNode,
    scenario: &'static str,
    items: usize,
    weights_mb: usize,
    steps: usize,
    opts: &ShardOptions,
) -> Result<Row, StruumError> {
    let mut row = Row {
        scenario,
        peers: node.node_count(),
        steps,
        wall: Duration::ZERO,
        sent: 0,
        received: 0,
        saved: 0,
        shards: 0,
        retries: 0,
    };
    let expected = selftest::expected(items);
    for _ in 0..steps {
        let task = task(items, weights_mb);
        let started = Instant::now();
        let (work, report): (_, ShardReport) = node.run_sharded_with_report(task, opts).await?;
        row.wall += started.elapsed();
        let out = work.into_cpu_task().expect("cpu result");
        assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>(), expected.as_slice());
        row.sent += report.bytes_sent;
        row.received += report.bytes_received;
        row.saved += report.bytes_saved;
        row.shards += report.shards;
        row.retries += report.retries;
    }
    Ok(row)
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let scenario = std::env::var("SCENARIO").unwrap_or_else(|_| "all".into());
    let items: usize = env_or("ITEMS", 1_000_000);
    let weights_mb: usize = env_or("WEIGHTS_MB", 8);
    let steps: usize = env_or("STEPS", 5);
    let opts = ShardOptions {
        shards_per_peer: env_or("SHARDS_PER_PEER", 2),
        blob_cache: env_or("BLOB_CACHE", 1u8) != 0,
        in_flight: env_or("IN_FLIGHT", 2),
        speculate: env_or("SPECULATE", 1u8) != 0,
        ..ShardOptions::default()
    };

    let mut node = BenchNode::with_id(UID::generate(), 0, 0, NodeBackend::Cpu, 1).await?;
    let peers = match std::env::var("RELAY_ADDR") {
        Ok(relay) => {
            let relay: SocketAddr = relay.parse().expect("invalid RELAY_ADDR");
            node.discover_peers(relay, env_or("MIN_PEERS", 2), Duration::from_secs(2)).await?
        }
        Err(_) => {
            let mut peers = Vec::new();
            for _ in 0..env_or("PEERS", 4usize) {
                peers.push(spawn_local_worker().await?);
            }
            for p in &peers {
                node.register_node(p.clone());
            }
            peers
        }
    };
    let connected = node.connect_to_peers(&peers).await;
    assert!(!connected.is_empty(), "no workers connected");

    println!(
        "{:<10} {:>5} {:>5} {:>10} {:>10} {:>10} {:>10} {:>7} {:>7}",
        "scenario", "peers", "steps", "wall ms", "MB sent", "MB recv", "MB saved", "shards", "retries"
    );
    let mut rows = Vec::new();
    if scenario == "all" || scenario == "split" {
        rows.push(run(&mut node, "split", items, 0, 1, &opts).await?);
        rows.last().unwrap().print();
    }
    if scenario == "all" || scenario == "broadcast" {
        rows.push(run(&mut node, "broadcast", items / 10, weights_mb, steps, &opts).await?);
        rows.last().unwrap().print();
    }

    if let Ok(path) = std::env::var("BENCH_JSON") {
        let json = format!("[{}]\n", rows.iter().map(Row::json).collect::<Vec<_>>().join(","));
        std::fs::write(&path, json).map_err(|e| StruumError::NotFound(format!("{path}: {e}")))?;
        println!("wrote {path}");
    }
    Ok(())
}
