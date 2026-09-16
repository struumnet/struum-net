use std::ops::{Deref, DerefMut};
use async_trait::async_trait;
use struum_node::node::Node;
use struum_types::{
    StruumError,
    network::{
        NetworkCoordinator, NodeBackend, NodeDetails, UID,
    },
};

/// Represents a Coordinator instance backed by an underlying `Node`.
/// Because `Node` maintains its own independent network map and coordination methods,
/// `Coordinator` provides a high-level coordination interface atop `Node`.
pub struct Coordinator<const BUF_SIZE: usize> {
    pub node: Node<BUF_SIZE>,
}

impl<const BUF_SIZE: usize> Deref for Coordinator<BUF_SIZE> {
    type Target = Node<BUF_SIZE>;

    fn deref(&self) -> &Self::Target {
        &self.node
    }
}

impl<const BUF_SIZE: usize> DerefMut for Coordinator<BUF_SIZE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.node
    }
}

impl<const BUF_SIZE: usize> Coordinator<BUF_SIZE> {
    /// Creates a new Coordinator backed by a local Node.
    pub async fn new(udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
        let node = Node::new(udp_port, tcp_port, NodeBackend::Cpu, 1).await?;
        Ok(Self { node })
    }

    /// Creates a new Coordinator with an explicit UID.
    pub async fn with_id(id: UID, udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
        let node = Node::with_id(id, udp_port, tcp_port, NodeBackend::Cpu, 1).await?;
        Ok(Self { node })
    }
}

#[async_trait]
impl<const BUF_SIZE: usize> NetworkCoordinator for Coordinator<BUF_SIZE> {
    async fn introduce_sibling_node(&mut self, node_id: &UID) -> Result<&NodeDetails, StruumError> {
        self.node.introduce_sibling_node(node_id).await
    }

    async fn introduce_nodes(&mut self, node_a_id: &UID, node_b_id: &UID) -> Result<(), StruumError> {
        self.node.introduce_nodes(node_a_id, node_b_id).await
    }

    async fn establish_information_exchange(&mut self) -> Result<(), StruumError> {
        self.node.establish_information_exchange().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use struum_types::network::NodeRole;

    #[tokio::test]
    async fn test_coordinator_registration_and_selection() {
        let mut coordinator = Coordinator::<1024>::new(38001, 38002)
            .await
            .expect("coordinator initialization failed");

        let node1 = NodeDetails {
            id: UID::new(1),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38011),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::OpenGL),
        };
        let node2 = NodeDetails {
            id: UID::new(2),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38021),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
        };

        coordinator.register_node(node1.clone());
        coordinator.register_node(node2.clone());

        assert_eq!(coordinator.node_count(), 2);
        assert_eq!(coordinator.get_node(&UID::new(1)), Some(&node1));

        let selected = coordinator.select_node();
        assert!(selected.is_some());

        let opengl_node = coordinator.select_node_by_backend(NodeBackend::OpenGL);
        assert_eq!(opengl_node.map(|n| n.id), Some(UID::new(1)));

        coordinator.deregister_node(&UID::new(1));
        assert_eq!(coordinator.get_node(&UID::new(1)), None);
        assert_eq!(coordinator.node_count(), 1);
    }
}
