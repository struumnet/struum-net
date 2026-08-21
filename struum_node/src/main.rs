pub mod node;
use struum_network::layer::NetworkLayer;
use struum_types::{StruumError, network::*};

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    let w_id = UID::new(0);
    let mut w1 = node::Node::<1024> {
        id: w_id,
        net: NetworkLayer::new(34255, 34265).await?,
        backend: NodeBackend::Cpu,
    };

    println!("WAITING FOR HELLO");
    let _ = w1.listen_hello().await?;
    println!("RECEIVED HELLO");
    let intro = w1.listen_introduction().await?;
    println!("RECEIVED INTRODUCTION!");
    let _ = w1.introduce_node(intro.ip).await?;
    println!("SENT INTRODUCTION!");
    Ok(())
}
