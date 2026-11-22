//! Sharded jobs over the QUIC transport: workers serve authenticated QUIC
//! peers with the same `PeerServer` as TCP, and the dispatcher is unchanged.

use std::net::SocketAddr;

use struum_cpu::selftest;
use struum_network::quic::QuicLayer;
use struum_node::{Node, ShardOptions};
use struum_types::network::{NodeBackend, UID};

fn local() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

/// A worker reachable over QUIC with `token`; returns its UDP address.
async fn spawn_quic_worker(id: u64, token: &'static [u8]) -> SocketAddr {
    let node = Node::<2048>::with_id(UID::new(id), 0, 0, NodeBackend::Cpu, 2).await.unwrap();
    let quic = QuicLayer::bind(local(), id, token).unwrap();
    let addr = quic.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok(peer) = quic.accept().await {
            tokio::spawn(node.server().serve(peer.conn.clone()));
            // QuicPeer handles keep the connection open; the PeerConnection clone does too.
            drop(peer);
        }
    });
    addr
}

#[tokio::test(flavor = "multi_thread")]
async fn cpu_job_runs_across_quic_workers() {
    let workers = [spawn_quic_worker(200, b"home").await, spawn_quic_worker(201, b"home").await];
    let mut node = Node::<2048>::with_id(UID::new(2), 0, 0, NodeBackend::Cpu, 1).await.unwrap();
    let quic = QuicLayer::bind(local(), 2, b"home").unwrap();

    let opts = ShardOptions { shards_per_peer: 3, ..ShardOptions::default() };
    let mut peers = Vec::new();
    for addr in workers {
        peers.push(quic.connect(addr).await.expect("authenticated"));
    }
    for (job, n) in [(1, 9_000), (2, 4_000)] {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        for p in &peers {
            tx.send(p.conn.clone()).unwrap();
        }
        drop(tx);
        let (out, report) = node.run_sharded_with_joins(selftest::task(n), rx, &opts).await.unwrap();
        assert_eq!(
            out.into_cpu_task().unwrap().get_buffer("y").unwrap().as_slice::<f32>(),
            selftest::expected(n).as_slice(),
            "job {job}"
        );
        assert_eq!(report.completed_by.len(), 2, "both QUIC workers took shards: {report:?}");
        assert_eq!(report.retries, 0);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn workers_with_another_token_are_unreachable() {
    let worker = spawn_quic_worker(210, b"theirs").await;
    let quic = QuicLayer::bind(local(), 3, b"mine").unwrap();
    assert!(quic.connect(worker).await.is_err());
}
