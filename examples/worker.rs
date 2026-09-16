use std::net::SocketAddr;
use struum_macros::gpu_type;
use struum_node::node;
use struum_types::{StruumError, network::*};

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

    let mut node = node::Node::<2048>::with_id(node_id, udp_port, tcp_port, NodeBackend::OpenGL, 1).await?;

    node.register_to_relay(relay_addr).await?;
    let _peer = node.listen_sibling_introduction().await?;
    let peer_conn_addr = node.accept_peer_connection().await?;

    loop {
        let (task_id, _sender_id, kernel) = match node.recv_kernel(&peer_conn_addr).await {
            Ok(data) => data,
            Err(_) => break,
        };

        let _job_id = node.submit_job(kernel).await;
        let completed = node.recv_completed_job().await.expect("Expected completed job");
        let result_kernel = node.get_job_result(&completed).await.expect("Expected result");

        node.send_task_result(peer_conn_addr, &task_id, &result_kernel).await?;
    }

    Ok(())
}
