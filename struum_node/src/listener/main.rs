use std::net::SocketAddr;
use struum_node::node;
use struum_types::{StruumError, network::*};

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let udp_port = std::env::var("NODE_UDP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(34255);
    let tcp_port = std::env::var("NODE_TCP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(34265);
    let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
        .parse()
        .expect("Invalid RELAY_ADDR");

    let mut node = node::Node::<2048>::new(
        udp_port,
        tcp_port,
        NodeBackend::Cpu,
        2,
    ).await?;

    println!("Node (listener) started!");
    println!(
        "We are: [UID] {} [IP] {} [UDP-PORT] {} [TCP-PORT] {}",
        node.id, node.net.ip, node.net.udp_port, node.net.tcp_port
    );
    println!("Registering to relay at {}...", relay_addr);

    node.register_to_relay(relay_addr).await?;
    println!("Registered! Waiting for relayed peer introductions from relay...");

    let peer = node.listen_sibling_introduction().await?;
    println!(
        "Successfully connected! Received peer introduction: UID {} at {}",
        peer.id, peer.ip
    );
    println!(
        "Current known peers in embedded coordinator: {:?}",
        node.coordinator.get_nodes()
    );

    Ok(())
}
