use struum_network::layer::NetworkLayer;
use struum_types::network::*;
use struum_node::node;


#[tokio::main]
async fn main() {
    let w_id = UID::new(0);
    let mut w1 = node::Node::<1024> {
        id: w_id,
        net: NetworkLayer { port: 34254 },
    };
    let _ = w1.listen_hello().await;
    print!("received hello message!");
}
