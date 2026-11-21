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
    let mut node = Node::<2048>::with_id(UID::new(id), 0, 0, NodeBackend::Cpu, 1)
        .await
        .expect("worker node");
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
