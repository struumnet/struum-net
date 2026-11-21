//! In-process tests for sharded execution: a requester node plus several
//! worker nodes on localhost, including peers that drop or hang.

use std::net::SocketAddr;
use std::time::Duration;

use struum_cpu::selftest;
use struum_kernel::Kernel;
use struum_node::{Node, ShardOptions};
use struum_types::Buffer;
use struum_types::network::{NodeBackend, NodeDetails, NodeRole, UID};

/// A real worker: serves every requester that connects.
async fn spawn_worker(id: u64) -> NodeDetails {
    spawn_worker_with_cache(id, None).await
}

/// A real worker, optionally with a blob cache of `cache_bytes`.
async fn spawn_worker_with_cache(id: u64, cache_bytes: Option<usize>) -> NodeDetails {
    let mut node = Node::<2048>::with_id(UID::new(id), 0, 0, NodeBackend::Cpu, 1)
        .await
        .expect("worker node");
    if let Some(bytes) = cache_bytes {
        node.blob_cache_bytes = bytes;
    }
    let details = node.details();
    tokio::spawn(async move {
        loop {
            let peer = node.accept_peer_connection().await.expect("accept");
            node.serve_peer(peer).await.expect("serve");
        }
    });
    details
}

#[derive(Clone, Copy)]
enum Faulty {
    /// Reads the first request, then closes the connection.
    Drops,
    /// Accepts the connection and never answers.
    Hangs,
}

/// A misbehaving worker that only speaks raw TCP.
async fn spawn_faulty(id: u64, mode: Faulty) -> NodeDetails {
    let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        match mode {
            Faulty::Drops => {
                use tokio::io::AsyncReadExt;
                let mut len = [0u8; 4];
                let _ = stream.read_exact(&mut len).await;
                drop(stream);
            }
            Faulty::Hangs => {
                tokio::time::sleep(Duration::from_secs(3600)).await;
                drop(stream);
            }
        }
    });
    NodeDetails {
        id: UID::new(id),
        ip: SocketAddr::new("127.0.0.1".parse().unwrap(), 1),
        role: NodeRole::NODE,
        backend: Some(NodeBackend::Cpu),
        tcp_port: Some(port),
    }
}

async fn requester(peers: &[NodeDetails]) -> Node<2048> {
    let mut node = Node::<2048>::with_id(UID::new(1), 0, 0, NodeBackend::Cpu, 1)
        .await
        .expect("requester node");
    for p in peers {
        node.register_node(p.clone());
    }
    let connected = node.connect_to_peers(peers).await;
    assert_eq!(connected.len(), peers.len(), "all peers should accept TCP");
    node
}

#[tokio::test(flavor = "multi_thread")]
async fn cpu_task_is_sharded_across_workers() {
    let mut peers = Vec::new();
    for id in 10..13 {
        peers.push(spawn_worker(id).await);
    }
    let mut node = requester(&peers).await;

    let n = 10_000;
    let (work, report) = node
        .run_sharded_with_report(selftest::task(n), &ShardOptions::default())
        .await
        .expect("sharded run");
    let out = work.into_cpu_task().expect("cpu result");

    assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>(), selftest::expected(n).as_slice());
    assert_eq!(report.shards, 6);
    assert_eq!(report.completed_by.values().sum::<usize>(), 6);
    assert!(report.completed_by.len() >= 2, "work should spread: {:?}", report.completed_by);
    // Per-shard bookkeeping: contiguous ranges covering 0..n, each with the peer that did it.
    assert_eq!(report.ranges.len(), 6);
    assert_eq!(report.ranges.first().unwrap().start, 0);
    assert_eq!(report.ranges.last().unwrap().end, n);
    assert_eq!(report.completed_on.len(), 6);
    for addr in &report.completed_on {
        assert!(report.completed_by.contains_key(addr));
    }
    assert!(report.retired.is_empty());
    // Transfer accounting: inputs and outputs both cross the wire.
    assert_eq!(report.round_trip.len(), 6);
    assert!(report.round_trip.iter().all(|d| !d.is_zero()));
    assert!(report.bytes_sent >= (n * 8) as u64, "{}", report.bytes_sent);
    assert!(report.bytes_received >= (n * 4) as u64, "{}", report.bytes_received);
}

#[tokio::test(flavor = "multi_thread")]
async fn shards_from_failed_peers_are_reassigned() {
    let peers = vec![
        spawn_worker(20).await,
        spawn_faulty(21, Faulty::Drops).await,
        spawn_faulty(22, Faulty::Hangs).await,
        spawn_worker(23).await,
    ];
    let mut node = requester(&peers).await;

    let opts = ShardOptions {
        shard_timeout: Duration::from_secs(2),
        ..ShardOptions::default()
    };
    let n = 4_000;
    let (work, report) = node
        .run_sharded_with_report(selftest::task(n), &opts)
        .await
        .expect("job should survive two bad peers");
    let out = work.into_cpu_task().unwrap();

    assert_eq!(out.get_buffer("y").unwrap().as_slice::<f32>(), selftest::expected(n).as_slice());
    assert_eq!(report.retired.len(), 2, "{report:?}");
    assert_eq!(report.retries, 2, "{report:?}");
    // Retired peers are forgotten for later jobs.
    assert_eq!(node.node_count(), 2);

    let again = node.run_sharded(selftest::task(100), &opts).await.expect("second run");
    assert_eq!(
        again.into_cpu_task().unwrap().get_buffer("y").unwrap().as_slice::<f32>(),
        selftest::expected(100).as_slice()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn task_errors_fail_the_job_without_retry() {
    let peers = vec![spawn_worker(30).await, spawn_worker(31).await];
    let mut node = requester(&peers).await;

    let err = node
        .run_sharded(selftest::task_with_entry("trap", 100), &ShardOptions::default())
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("trapped"), "{err}");
    // Deterministic task errors don't retire healthy peers.
    assert_eq!(node.node_count(), 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn gpu_work_needs_opengl_peers() {
    let peers = vec![spawn_worker(40).await];
    let mut node = requester(&peers).await;

    let mut kernel = Kernel::new("void compute(uint i) { y[i] = x[i]; }", "compute");
    kernel.add_buffer::<f32>(Buffer::new("x", &[1.0f32, 2.0]));
    kernel.add_buffer::<f32>(Buffer::empty::<f32>("y", 2));
    kernel.set_work_buffer("x");
    kernel.pack().unwrap();

    let err = node.run_sharded(kernel, &ShardOptions::default()).await.unwrap_err().to_string();
    assert!(err.contains("no connected peers"), "{err}");
}

#[tokio::test(flavor = "multi_thread")]
async fn stats_count_contributions_both_ways() {
    // A worker whose stats we can read after it served.
    let mut worker = Node::<2048>::with_id(UID::new(50), 0, 0, NodeBackend::Cpu, 1).await.unwrap();
    let worker_stats = worker.stats();
    let details = worker.details();
    tokio::spawn(async move {
        loop {
            let peer = worker.accept_peer_connection().await.expect("accept");
            worker.serve_peer(peer).await.expect("serve");
        }
    });
    let mut node = requester(&[details]).await;

    let opts = ShardOptions { shards_per_peer: 3, ..ShardOptions::default() };
    let (_, report) = node.run_sharded_with_report(selftest::task(300), &opts).await.unwrap();

    let mine = node.stats().snapshot();
    assert_eq!(mine.jobs, 1);
    assert_eq!(mine.used.shards, 3);
    assert_eq!(mine.used.bytes_out, report.bytes_sent);
    assert_eq!(mine.used_by_peer.len(), 1);
    assert_eq!(mine.served.shards, 0, "the requester ran nothing for others");

    let theirs = worker_stats.snapshot();
    assert_eq!(theirs.served.shards, 3);
    assert_eq!(theirs.served.bytes_in, report.bytes_sent);
    assert_eq!(theirs.served.bytes_out, report.bytes_received);
    assert!(!theirs.working);
}

/// Selftest task plus a 64 KiB broadcast buffer, like weights in training.
fn task_with_weights(n: usize, fill: f32) -> struum_cpu::CpuTask {
    let mut task = selftest::task(n);
    task.add_buffer(Buffer::new("weights", &vec![fill; 16_384]));
    task
}

#[tokio::test(flavor = "multi_thread")]
async fn repeated_blobs_are_sent_once_per_peer() {
    let peers = vec![spawn_worker(60).await, spawn_worker(61).await];
    let mut node = requester(&peers).await;
    let opts = ShardOptions { shards_per_peer: 3, ..ShardOptions::default() };
    let weights = 16_384 * 4;
    let n = 600;

    let (first, r1) = node.run_sharded_with_report(task_with_weights(n, 1.0), &opts).await.unwrap();
    // Each peer that took work needed the weights once; its other shards refer to them.
    let full_sends = r1.completed_by.len() as u64;
    assert_eq!(r1.bytes_saved, (6 - full_sends) * weights as u64, "{r1:?}");
    let (second, r2) = node.run_sharded_with_report(task_with_weights(n, 1.0), &opts).await.unwrap();
    // Same weights next job: no shard carries them.
    assert_eq!(r2.bytes_saved, 6 * weights as u64, "{r2:?}");
    assert_eq!(r2.bytes_sent + r2.bytes_saved, r1.bytes_sent + r1.bytes_saved);
    assert_eq!(r2.cache_misses, 0);

    // Identical results to a run without the cache.
    let plain = ShardOptions { blob_cache: false, ..opts.clone() };
    let (reference, r3) = node.run_sharded_with_report(task_with_weights(n, 1.0), &plain).await.unwrap();
    assert_eq!(r3.bytes_saved, 0);
    for out in [first, second] {
        assert_eq!(out.to_bytes().unwrap(), reference.to_bytes().unwrap());
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn evicted_blobs_are_resent_without_retiring_the_peer() {
    // Room for less than one weights buffer: the worker never keeps them.
    let peers = vec![spawn_worker_with_cache(70, Some(0)).await];
    let mut node = requester(&peers).await;
    let opts = ShardOptions { shards_per_peer: 2, ..ShardOptions::default() };

    let (out, report) = node.run_sharded_with_report(task_with_weights(200, 2.0), &opts).await.unwrap();
    assert!(report.cache_misses >= 1, "{report:?}");
    assert_eq!(report.retries, 0);
    assert!(report.retired.is_empty());
    assert_eq!(
        out.into_cpu_task().unwrap().get_buffer("y").unwrap().as_slice::<f32>(),
        selftest::expected(200).as_slice()
    );
}
