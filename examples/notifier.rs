use struum_network::layer::NetworkLayer;
use struum_node::node;
use struum_types::network::*;

#[tokio::main]
async fn main() {
    let w_id = UID::new(0);
    let mut w1 = node::Node::<1024> {
        id: w_id,
        net: NetworkLayer { port: 34254 },
    };
    let _ = w1.notify_network().await;
    print!("sent hello message!");
}
