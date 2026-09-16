use std::collections::HashMap;
use async_trait::async_trait;
use struum_network::layer::NetworkLayer;
use struum_types::{
    StruumError,
    network::{
        IntroductionPacket, NetworkCoordinator, NodeBackend, NodeDetails, UID, UdpPacket,
    },
};

/// Represents the Coordinator for coordinating communication between nodes in the network.
pub struct Coordinator<const BUF_SIZE: usize> {
    /// Unique identifier for the coordinator node
    pub id: UID,
    /// Network layer used for inter-node communication
    pub net: NetworkLayer<BUF_SIZE>,
    /// Registered nodes in the network
    pub nodes: HashMap<UID, NodeDetails>,
    /// Cursor for round-robin node selection
    next_node: usize,
}

impl<const BUF_SIZE: usize> Coordinator<BUF_SIZE> {
    pub async fn new(udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
        Ok(Self {
            id: UID::new(0),
            net: NetworkLayer::new(udp_port, tcp_port).await?,
            nodes: HashMap::default(),
            next_node: 0,
        })
    }

    /// Registers a node as available in the network.
    pub fn register_node(&mut self, details: NodeDetails) {
        self.nodes.insert(details.id, details);
    }

    /// Removes a node from the network (e.g. on disconnect).
    pub fn deregister_node(&mut self, id: &UID) {
        self.nodes.remove(id);
    }

    /// Returns a reference to a registered node by UID.
    pub fn get_node(&self, id: &UID) -> Option<&NodeDetails> {
        self.nodes.get(id)
    }

    /// Returns all registered nodes.
    pub fn get_nodes(&self) -> Vec<NodeDetails> {
        self.nodes.values().cloned().collect()
    }

    /// Number of registered nodes in the network.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Round-robin pick among registered nodes to balance work across nodes.
    pub fn select_node(&mut self) -> Option<NodeDetails> {
        if self.nodes.is_empty() {
            return None;
        }
        let ids: Vec<UID> = self.nodes.keys().copied().collect();
        let id = ids[self.next_node % ids.len()];
        self.next_node = self.next_node.wrapping_add(1);
        self.nodes.get(&id).cloned()
    }

    /// Selects a registered node matching the requested computation backend.
    pub fn select_node_by_backend(&mut self, backend: NodeBackend) -> Option<NodeDetails> {
        self.nodes
            .values()
            .find(|node| node.backend == Some(backend))
            .cloned()
    }

    /// Listens for a node introduction packet via UDP and registers it.
    pub async fn listen_for_node_registration(&mut self) -> Result<NodeDetails, StruumError> {
        let (intro, _src) = self.net.listen_udp::<IntroductionPacket>().await?;
        let details = NodeDetails::from(intro);
        self.register_node(details.clone());
        Ok(details)
    }

    /// Introduces two sibling nodes to each other so they can establish direct communication.
    pub async fn introduce_nodes(&mut self, node_a_id: &UID, node_b_id: &UID) -> Result<(), StruumError> {
        let node_a = self
            .nodes
            .get(node_a_id)
            .cloned()
            .ok_or_else(|| StruumError::NotFound(format!("Node {} not found", node_a_id)))?;
        let node_b = self
            .nodes
            .get(node_b_id)
            .cloned()
            .ok_or_else(|| StruumError::NotFound(format!("Node {} not found", node_b_id)))?;

        // Send Node B's details to Node A
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    id: node_b.id,
                    ip: node_b.ip,
                    role: node_b.role,
                    backend: node_b.backend,
                }),
                node_a.ip,
            )
            .await?;

        // Send Node A's details to Node B
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    id: node_a.id,
                    ip: node_a.ip,
                    role: node_a.role,
                    backend: node_a.backend,
                }),
                node_b.ip,
            )
            .await?;

        Ok(())
    }
}

#[async_trait]
impl<const BUF_SIZE: usize> NetworkCoordinator for Coordinator<BUF_SIZE> {
    /// Looks up a registered node by ID.
    async fn introduce_sibling_node(&mut self, node_id: &UID) -> Result<&NodeDetails, StruumError> {
        self.nodes.get(node_id).ok_or_else(|| {
            StruumError::NotFound("Node not found in registered network!".to_string())
        })
    }

    /// Introduces two sibling nodes to each other so they can establish direct communication.
    async fn introduce_nodes(&mut self, node_a_id: &UID, node_b_id: &UID) -> Result<(), StruumError> {
        self.introduce_nodes(node_a_id, node_b_id).await
    }

    /// Cross-introduces all registered sibling nodes to each other in the network (mesh coordination).
    async fn establish_information_exchange(&mut self) -> Result<(), StruumError> {
        let node_list: Vec<NodeDetails> = self.nodes.values().cloned().collect();
        for i in 0..node_list.len() {
            for j in (i + 1)..node_list.len() {
                let a = &node_list[i];
                let b = &node_list[j];
                self.introduce_nodes(&a.id, &b.id).await?;
            }
        }
        Ok(())
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
        assert_eq!(coordinator.node_count(), 1);
        assert_eq!(coordinator.get_node(&UID::new(1)), None);
    }
}

