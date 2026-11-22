//! `serving` the way `struum_worker` uses it: prepare QUIC, register with the
//! relay through the node itself, then serve. With a token, the relay must not
//! hand out the TCP port, and requesters reach the worker over QUIC.

use std::net::SocketAddr;
use std::time::Duration;

use struum_cpu::selftest;
use struum_node::{Node, ShardOptions, serving};
use struum_relay::RelayServer;
use struum_types::network::{NodeBackend, UID};

async fn relay() -> SocketAddr {
    relay_with_ttl(struum_relay::DEFAULT_TTL).await
}

async fn relay_with_ttl(ttl: Duration) -> SocketAddr {
    let mut relay = RelayServer::<2048>::new(0, 0).await.unwrap().with_ttl(ttl);
    let addr = SocketAddr::from(([127, 0, 0, 1], relay.net.udp_port));
    tokio::spawn(async move { relay.run().await });
    addr
}

/// A worker set up like `struum_worker`: returns its id and what it listens on.
async fn spawn_worker(relay: SocketAddr, token: &[u8]) -> (UID, serving::Listening) {
    spawn_worker_with_heartbeat(relay, token, serving::HEARTBEAT).await
}

async fn spawn_worker_with_heartbeat(relay: SocketAddr, token: &[u8], beat: Duration) -> (UID, serving::Listening) {
    let mut node = Node::<2048>::with_id(UID::generate(), 0, 0, NodeBackend::Cpu, 2).await.unwrap();
    let listening = serving::prepare(&mut node, 0, token).unwrap();
    let id = node.id;
    tokio::spawn(async move {
        node.long_poll_relay_with_interval(relay, Duration::from_millis(200)).await.unwrap();
        serving::spawn_with_heartbeat(node, listening, beat);
    });
    (id, listening)
}

async fn requester(relay: SocketAddr, token: &[u8]) -> (Node<2048>, Vec<struum_types::network::NodeDetails>) {
    let mut node = Node::<2048>::with_id(UID::generate(), 0, 0, NodeBackend::Cpu, 1).await.unwrap();
    node.enable_quic(0, token).unwrap();
    let peers = tokio::time::timeout(Duration::from_secs(10), node.discover_peers(relay, 1, Duration::from_millis(300)))
        .await
        .expect("relay introduces the worker")
        .unwrap();
    (node, peers)
}

#[tokio::test(flavor = "multi_thread")]
async fn token_worker_advertises_quic_only_and_serves_over_it() {
    const TOKEN: &[u8] = b"serving-test";
    let relay = relay().await;
    let (id, listening) = spawn_worker(relay, TOKEN).await;
    assert_eq!(listening.tcp_port, None);
    assert!(listening.authenticated);

    let (mut node, peers) = requester(relay, TOKEN).await;
    let worker = peers.iter().find(|p| p.id == id).expect("worker introduced");
    assert_eq!(worker.tcp_port, None, "TCP is off, so it isn't advertised");
    assert_eq!(worker.quic_port, Some(listening.quic_port));

    assert_eq!(node.connect_to_peers(&peers).await.len(), 1);
    let opts = ShardOptions { shards_per_peer: 3, ..ShardOptions::default() };
    let out = node.run_sharded(selftest::task(3000), &opts).await.unwrap();
    assert_eq!(
        out.into_cpu_task().unwrap().get_buffer("y").unwrap().as_slice::<f32>(),
        selftest::expected(3000).as_slice()
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn open_worker_advertises_tcp_too() {
    let relay = relay().await;
    let (id, listening) = spawn_worker(relay, b"").await;
    assert!(!listening.authenticated);
    assert!(listening.tcp_port.is_some());

    let (_node, peers) = requester(relay, b"").await;
    let worker = peers.iter().find(|p| p.id == id).expect("worker introduced");
    assert_eq!(worker.tcp_port, listening.tcp_port);
    assert_eq!(worker.quic_port, Some(listening.quic_port));
}

#[tokio::test(flavor = "multi_thread")]
async fn serving_worker_stays_registered_past_the_ttl() {
    let ttl = Duration::from_millis(500);
    let relay = relay_with_ttl(ttl).await;
    let (id, _) = spawn_worker_with_heartbeat(relay, b"", Duration::from_millis(100)).await;

    // The first requester gets the worker past long_poll_relay and serving.
    let (_first, peers) = requester(relay, b"").await;
    assert!(peers.iter().any(|p| p.id == id));

    // Long after the TTL, a new requester still finds it.
    tokio::time::sleep(ttl * 3).await;
    let (_later, peers) = requester(relay, b"").await;
    assert!(peers.iter().any(|p| p.id == id), "heartbeat kept the worker registered: {peers:?}");
}
