pub mod node;
use struum_network::layer::NetworkLayer;
use struum_types::{StruumError, network::*};


#[tokio::main]
async fn main() ->Result<(),StruumError> {
    let w_id = UID::new(0);
    let mut w1 = node::Node::<1024> {
        id: w_id,
        net: NetworkLayer { port: 34254 },
    };
    let _ = w1.listen_hello().await?;
    println!("Waiting for Hello!");
    let _ = w1.notify_network().await?;
    println!("Hello sent!");
    Ok(())
}
