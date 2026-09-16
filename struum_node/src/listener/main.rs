use struum_node::node;
use struum_types::{StruumError, network::*};

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    let mut w1 = node::Node::<2048>::new(
        34255,
        34265,
        NodeBackend::Cpu,
        2,
    ).await?;

    println!("Started Network layer listening!");
    println!("We are: \n[IP]{}\n[UDP-PORT]{}\n[TCP-PORT]{}", w1.net.ip, w1.net.udp_port, w1.net.tcp_port);

    println!("Waiting for introduction!");
    let _ = w1.notify_network().await?;
    let _ = w1.listen_introduction().await?;
    Ok(())
}
