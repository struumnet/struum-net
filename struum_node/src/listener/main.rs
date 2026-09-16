use std::net::SocketAddr;
use struum_macros::gpu_type;
use struum_node::node;
use struum_types::{StruumError, network::*};

#[gpu_type]
#[derive(Debug, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

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

    let node_id = std::env::var("NODE_ID")
        .ok()
        .and_then(|p| p.parse().ok())
        .map(UID::new)
        .unwrap_or(UID::new(1));

    let mut node = node::Node::<2048>::with_id(
        node_id,
        udp_port,
        tcp_port,
        NodeBackend::OpenGL,
        2,
    ).await?;

    log::info!("Node (listener) started!");
    log::info!(
        "We are: [UID] {} [IP] {} [UDP-PORT] {} [TCP-PORT] {}",
        node.id, node.net.ip, node.net.udp_port, node.net.tcp_port
    );
    log::info!("Sending registration request to relay at {}...", relay_addr);

    node.register_to_relay(relay_addr).await?;
    log::info!("Waiting for relayed peer introductions from relay...");

    let peer = node.listen_sibling_introduction().await?;
    log::info!(
        "Successfully connected! Received peer introduction: UID {} at {}",
        peer.id, peer.ip
    );
    println!(
        "Current known peers in embedded coordinator: {:?}",
        node.coordinator.get_nodes()
    );

    println!("Waiting for incoming TCP connection from peer...");
    let peer_conn_addr = node.accept_peer_connection().await?;
    println!("Accepted TCP connection from peer at {}!", peer_conn_addr);

    println!("Waiting for kernel over TCP...");
    let (task_id, sender_id, received_kernel) = node.recv_kernel(&peer_conn_addr).await?;
    println!(
        "Received kernel: [Task ID] {} [Sender UID] {} [Bindings Count] {}",
        task_id,
        sender_id,
        received_kernel.get_buffer_bindings().len()
    );

    println!("Submitting received kernel to local scheduler for OpenGL computation...");
    let _job_id = node.submit_job(received_kernel).await;
    let completed = node.recv_completed_job().await.expect("Expected completed job ID");
    println!("Job {} computation completed on local worker threads!", completed);

    let result_kernel = node.get_job_result(&completed).await.expect("Expected result kernel");

    if let Some(output) = result_kernel.get_buffer("output") {
        let slice = output.as_slice::<Vec2>();
        println!("Computed output buffer: {:?}", slice);
    }

    println!("Sending computation result back to peer over TCP...");
    node.send_task_result(peer_conn_addr, &task_id, &result_kernel).await?;
    println!("Result sent successfully!");

    Ok(())
}
