//! End-to-end relay tests: nodes register over UDP and get introduced to each other.

use std::net::SocketAddr;
use std::time::Duration;

use struum_network::layer::NetworkLayer;
use struum_relay::RelayServer;
use struum_types::network::{IntroductionPacket, NodeBackend, NodeRole, UID, UdpPacket};
use tokio::time::timeout;

const WAIT: Duration = Duration::from_secs(5);

fn local(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

struct TestNode {
    id: UID,
    net: NetworkLayer<1024>,
}

impl TestNode {
    async fn new() -> Self {
        Self { id: UID::generate(), net: NetworkLayer::new(0, 0).await.unwrap() }
    }

    /// Registers with the relay the way real nodes do: unspecified IP, own UDP port.
    async fn register(&mut self, relay_port: u16) {
        let intro = IntroductionPacket {
            id: self.id,
            ip: SocketAddr::from(([0, 0, 0, 0], self.net.udp_port)),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
            tcp_port: Some(self.net.tcp_port),
            quic_port: None,
        };
        self.net
            .send_udp_data(UdpPacket::INTRODUCTION(intro), local(relay_port))
            .await
            .unwrap();
    }

    async fn next_intro(&mut self) -> IntroductionPacket {
        timeout(WAIT, self.net.listen_udp::<IntroductionPacket>())
            .await
            .expect("introduction should arrive")
            .unwrap()
            .0
    }
}

async fn relay() -> RelayServer<1024> {
    RelayServer::new(0, 0).await.expect("relay")
}

#[tokio::test]
async fn unspecified_ip_is_replaced_by_the_source_address() {
    let mut relay = relay().await;
    let mut node = TestNode::new().await;

    node.register(relay.net.udp_port).await;
    let details = timeout(WAIT, relay.listen_and_relay_once()).await.unwrap().unwrap();

    assert_eq!(details.id, node.id);
    assert_eq!(details.ip, local(node.net.udp_port));
    assert_eq!(details.tcp_addr(), Some(local(node.net.tcp_port)));
    assert_eq!(relay.get_node(&node.id), Some(&details));
}

#[tokio::test]
async fn new_nodes_are_introduced_to_existing_ones_and_back() {
    let mut relay = relay().await;
    let mut a = TestNode::new().await;
    let mut b = TestNode::new().await;

    a.register(relay.net.udp_port).await;
    timeout(WAIT, relay.listen_and_relay_once()).await.unwrap().unwrap();
    b.register(relay.net.udp_port).await;
    timeout(WAIT, relay.listen_and_relay_once()).await.unwrap().unwrap();

    let a_learns = a.next_intro().await;
    assert_eq!(a_learns.id, b.id);
    assert_eq!(a_learns.ip, local(b.net.udp_port));
    assert_eq!(a_learns.tcp_port, Some(b.net.tcp_port));

    let b_learns = b.next_intro().await;
    assert_eq!(b_learns.id, a.id);
    assert_eq!(b_learns.ip, local(a.net.udp_port));
    assert_eq!(relay.registered_count(), 2);
}

#[tokio::test]
async fn re_registering_refreshes_without_duplicating() {
    let mut relay = relay().await;
    let mut a = TestNode::new().await;

    for _ in 0..3 {
        a.register(relay.net.udp_port).await;
        timeout(WAIT, relay.listen_and_relay_once()).await.unwrap().unwrap();
    }
    assert_eq!(relay.registered_count(), 1);

    // A lone node is never introduced to itself.
    let self_intro = timeout(Duration::from_millis(200), a.net.listen_udp::<IntroductionPacket>()).await;
    assert!(self_intro.is_err(), "unexpected introduction: {self_intro:?}");
}

#[tokio::test]
async fn garbage_registration_is_an_error_and_registers_nothing() {
    let mut relay = relay().await;
    let sender = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    sender.send_to(b"not a packet", local(relay.net.udp_port)).await.unwrap();

    assert!(timeout(WAIT, relay.listen_and_relay_once()).await.unwrap().is_err());
    assert_eq!(relay.registered_count(), 0);
}
