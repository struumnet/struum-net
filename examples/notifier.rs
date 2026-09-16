use struum_node::node;
use struum_types::network::*;

#[tokio::main]
async fn main() {
    let w_id = UID::new(0);
    let mut w1 = node::Node::<1024>::new(w_id, 34254, 34264, NodeBackend::Cpu, 2)
        .await
        .expect("Failed to create node");
    let _ = w1.notify_network().await;
    println!("sent hello message!");
}
