use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use struum_node::Node;
use struum_relay::RelayServer;
use struum_types::network::NodeBackend;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let relay_udp_port = 39101;
    let relay_tcp_port = 39102;
    let relay_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), relay_udp_port);

    println!("--- 1. Starting Relay Server on port {} ---", relay_udp_port);
    let mut relay = RelayServer::<2048>::new(relay_udp_port, relay_tcp_port).await?;

    // Spawn relay listening loop in background
    tokio::spawn(async move {
        println!("Relay server loop running...");
        let _ = relay.run().await;
    });

    // Give relay socket time to bind
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut node1 = Node::<2048>::new(39111, 39112, NodeBackend::OpenGL, 2).await?;
    let node1_id = node1.id;

    let mut node2 = Node::<2048>::new(39121, 39122, NodeBackend::Cpu, 2).await?;
    let node2_id = node2.id;

    // Spawn listeners on nodes to listen for relayed introductions from the relay
    let listen_handle1 = tokio::spawn(async move {
        let peer = node1.listen_sibling_introduction().await?;
        Ok::<Node<2048>, struum_types::StruumError>(node1)
    });

    let listen_handle2 = tokio::spawn(async move {
        let peer = node2.listen_sibling_introduction().await?;
        Ok::<Node<2048>, struum_types::StruumError>(node2)
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut temp_node1 = Node::<2048>::with_id(node1_id, 39131, 39132, NodeBackend::OpenGL, 1).await?;
    temp_node1.net.udp_port = 39111;
    temp_node1.register_to_relay(relay_addr).await?;

    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut temp_node2 = Node::<2048>::with_id(node2_id, 39141, 39142, NodeBackend::Cpu, 1).await?;
    temp_node2.net.udp_port = 39121;
    temp_node2.register_to_relay(relay_addr).await?;

    let (node1_res, node2_res) = tokio::join!(listen_handle1, listen_handle2);
    let node1 = node1_res??;
    let node2 = node2_res??;


    assert_eq!(node1.coordinator.node_count(), 1);
    assert_eq!(node2.coordinator.node_count(), 1);
    assert!(node1.coordinator.get_node(&node2_id).is_some());
    assert!(node2.coordinator.get_node(&node1_id).is_some());

    Ok(())
}
