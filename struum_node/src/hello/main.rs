use std::net::SocketAddr;
use struum_kernel::Kernel;
use struum_macros::gpu_type;
use struum_node::node;
use struum_types::{Buffer, StruumError, network::*};

#[gpu_type]
#[derive(Debug, PartialEq)]
struct Vec2 {
    x: f32,
    y: f32,
}

#[gpu_type]
#[derive(Debug, PartialEq)]
struct Param {
    m: f32,
    c: f32,
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let udp_port = std::env::var("NODE_UDP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(34256);
    let tcp_port = std::env::var("NODE_TCP_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(34266);
    let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
        .parse()
        .expect("Invalid RELAY_ADDR");

    let node_id = std::env::var("NODE_ID")
        .ok()
        .and_then(|p| p.parse().ok())
        .map(UID::new)
        .unwrap_or(UID::new(2));

    let mut node = node::Node::<2048>::with_id(
        node_id,
        udp_port,
        tcp_port,
        NodeBackend::Cpu,
        2,
    ).await?;

    println!("Node (hello) started!");
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

    let peer_tcp = peer
        .tcp_addr()
        .expect("Peer did not provide TCP address");
    println!("Connecting to peer over TCP at {}...", peer_tcp);

    node.connect_to_peer_addr(peer_tcp).await?;
    println!("Connected to peer over TCP!");

    // Construct computational task (kernel with GLSL source and buffer bindings)
    let source = r#"
        void compute(uint i) {
            Param p = param[0];
            Vec2 x = pos[i];
            output[i] = Vec2(
                p.m * x.x,
                p.c * x.y
            );
        }
    "#;

    let mut kernel = Kernel::new(source, "compute");
    kernel.add_buffer::<Param>(Buffer::new("param", &[Param { m: 3.0, c: 4.0 }]));
    kernel.add_buffer::<Vec2>(Buffer::new(
        "pos",
        &[Vec2 { x: 2.0, y: 5.0 }, Vec2 { x: 4.0, y: 10.0 }],
    ));
    kernel.add_buffer::<Vec2>(Buffer::empty::<Vec2>("output", 2));
    kernel.set_work_buffer("pos");
    kernel.pack()?;

    println!("Sending kernel to peer over TCP...");
    node.send_kernel(peer_tcp, "task-1001", &kernel).await?;
    println!("Kernel sent successfully! Awaiting computation result...");

    let (job_id, sender_id, result_kernel) = node.recv_task_result(&peer_tcp).await?;
    println!(
        "Received computation result for job '{}' from peer {}!",
        job_id, sender_id
    );

    if let Some(output) = result_kernel.get_buffer("output") {
        let slice = output.as_slice::<Vec2>();
        println!("Computed output buffer: {:?}", slice);
    }

    Ok(())
}
