use std::net::SocketAddr;
use struum_kernel::Kernel;
use struum_macros::gpu_type;
use struum_node::node;
use struum_types::{Buffer, StruumError, network::*};

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

fn build_gradient_kernel(params: &Param, x: &[f32], y: &[f32]) -> Kernel {
    let source = r#"
        void compute(uint i) {
            Param p = param[0];

            float prediction = p.m * x[i] + p.c;
            float error = prediction - y[i];

            gradients[i] = Gradient(
                2.0 * x[i] * error,
                2.0 * error
            );
        }
    "#;

    let mut kernel = Kernel::new(source, "compute");
    kernel.add_buffer::<Param>(Buffer::new("param", &[Param { m: params.m, c: params.c }]));
    kernel.add_buffer::<f32>(Buffer::new("x", x));
    kernel.add_buffer::<f32>(Buffer::new("y", y));
    kernel.add_buffer::<Gradient>(Buffer::empty::<Gradient>("gradients", x.len()));
    kernel.set_work_buffer("x");
    kernel.pack().expect("Failed to pack kernel");
    kernel
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

    let mut node = node::Node::<2048>::with_id(node_id, udp_port, tcp_port, NodeBackend::Cpu, 1).await?;

    println!("Requester node started!");
    println!(
        "We are: [UID] {} [IP] {} [UDP-PORT] {} [TCP-PORT] {}",
        node.id, node.net.ip, node.net.udp_port, node.net.tcp_port
    );
    println!("Registering to relay at {}...", relay_addr);

    node.register_to_relay(relay_addr).await?;
    println!("Registered! Waiting for peer introduction from relay...");

    let peer = node.listen_sibling_introduction().await?;
    println!("Received peer introduction: UID {} at {}", peer.id, peer.ip);

    let peer_tcp = peer.tcp_addr().expect("Peer did not provide TCP address");
    println!("Connecting to worker over TCP at {}...", peer_tcp);

    node.connect_to_peer_addr(peer_tcp).await?;
    println!("Connected to worker over TCP!");

    let x: Vec<f32> = (0..5).map(|i| i as f32).collect();
    let y: Vec<f32> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

    let mut params = Param { m: 0.0, c: 0.0 };

    println!("Computation..");
    for iteration in 0..1000 {
        let kernel = build_gradient_kernel(&params, &x, &y);
        let task_id = format!("grad-iter-{}", iteration);

        node.send_kernel(peer_tcp, &task_id, &kernel).await?;
        let (_ret_id, _sender, result_kernel) = node.recv_task_result(&peer_tcp).await?;

        let grad_buf = result_kernel.get_buffer("gradients").expect("Expected gradients buffer");
        let gradients = grad_buf.as_slice::<Gradient>();

        let mut dm = 0.0_f32;
        let mut dc = 0.0_f32;
        for g in gradients {
            dm += g.dm;
            dc += g.dc;
        }
        dm /= x.len() as f32;
        dc /= x.len() as f32;

        params.m -= 0.01 * dm;
        params.c -= 0.01 * dc;

        if iteration % 100 == 0 {
            println!(
                "iteration {}: m={}, c={}",
                iteration, params.m, params.c,
            );
        }
    }

    println!();
    println!("Final model:");
    println!("m = {}", params.m);
    println!("c = {}", params.c);

    Ok(())
}
