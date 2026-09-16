use struum_node::node;
use struum_types::network::*;

#[tokio::main]
async fn main() {
    let mut w1 = node::Node::<1024>::new(34254, 34264, NodeBackend::Cpu, 2)
        .await
        .expect("Failed to create node");
    let _ = w1.listen_hello().await;
    println!("received hello message!");
}
