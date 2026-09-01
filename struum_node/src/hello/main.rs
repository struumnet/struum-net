use struum_node::node;
use struum_network::layer::NetworkLayer;
use struum_types::{StruumError, network::*};

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    let w_id = UID::new(1);
    let mut w1 = node::Node::<2048> {
        id: w_id,
        net: NetworkLayer::new(
            34255,
            34265
            ).await?,
        backend: NodeBackend::Cpu,
    };
    println!("Started Network layer listening!");
    println!("We are: \n[IP]{}\n[UDP-PORT]{}\n[TCP-PORT]{}",w1.net.ip,w1.net.udp_port,w1.net.tcp_port);

    println!("Introducing to network!");
    let ip = w1.listen_hello().await?;
    let intro = w1.introduce(ip).await?;
    Ok(())
}
