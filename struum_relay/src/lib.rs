use std::collections::HashMap;
use std::net::SocketAddr;
use struum_network::layer::NetworkLayer;
use struum_types::StruumError;
use struum_types::network::{IntroductionPacket, NodeDetails, UID, UdpPacket};

/// RelayServer coordinates and relays node IP registrations.
/// Nodes register their IP and role by sending an IntroductionPacket to the relay.
/// The relay records the node and relays the newly registered IP to all existing nodes,
/// as well as relaying all existing nodes' IPs to the new node.
pub struct RelayServer<const BUF_SIZE: usize> {
    pub id: UID,
    pub net: NetworkLayer<BUF_SIZE>,
    pub registered_nodes: HashMap<UID, NodeDetails>,
}

impl<const BUF_SIZE: usize> RelayServer<BUF_SIZE> {
    /// Creates a new RelayServer listening on the specified UDP and TCP ports.
    pub async fn new(udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
        let net = NetworkLayer::new(udp_port, tcp_port).await?;
        Ok(Self {
            id: UID::generate(),
            net,
            registered_nodes: HashMap::default(),
        })
    }

    /// Number of registered nodes currently stored in the relay.
    pub fn registered_count(&self) -> usize {
        self.registered_nodes.len()
    }

    /// Returns a registered node by UID.
    pub fn get_node(&self, id: &UID) -> Option<&NodeDetails> {
        self.registered_nodes.get(id)
    }

    /// Registers a node and immediately relays its IP to all other registered nodes,
    /// while sending existing nodes' IPs to the new node.
    pub async fn register_and_relay_node(&mut self, details: NodeDetails) -> Result<(), StruumError> {
        let is_new = !self.registered_nodes.contains_key(&details.id);
        if is_new {
            log::info!(
                "Relay registered node {} at IP {} (backend: {:?})",
                details.id,
                details.ip,
                details.backend
            );
        } else {
            log::debug!(
                "Relay refreshed registration for node {} at IP {}",
                details.id,
                details.ip
            );
        }

        // Relay between the newly registered node and all existing nodes
        for (existing_id, existing_node) in &self.registered_nodes {
            if *existing_id != details.id {
                log::info!(
                    "Relaying IPs: introducing {} ({}) <---> {} ({})",
                    details.id,
                    details.ip,
                    existing_node.id,
                    existing_node.ip
                );

                // Relay new node's IP to existing node
                self.net
                    .send_udp_data(
                        UdpPacket::INTRODUCTION(IntroductionPacket {
                            id: details.id,
                            ip: details.ip,
                            role: details.role,
                            backend: details.backend,
                            tcp_port: details.tcp_port,
                        }),
                        existing_node.ip,
                    )
                    .await?;

                // Relay existing node's IP to new node
                self.net
                    .send_udp_data(
                        UdpPacket::INTRODUCTION(IntroductionPacket {
                            id: existing_node.id,
                            ip: existing_node.ip,
                            role: existing_node.role,
                            backend: existing_node.backend,
                            tcp_port: existing_node.tcp_port,
                        }),
                        details.ip,
                    )
                    .await?;
            }
        }

        self.registered_nodes.insert(details.id, details);
        Ok(())
    }

    /// Listens for a single incoming node registration packet and relays it.
    pub async fn listen_and_relay_once(&mut self) -> Result<NodeDetails, StruumError> {
        let (intro, src) = self.net.listen_udp::<IntroductionPacket>().await?;
        let mut details = NodeDetails::from(intro);
        // If the reporting node did not specify its public IP or specified 0.0.0.0, use the UDP source socket address
        if details.ip.ip().is_unspecified() {
            details.ip = SocketAddr::new(src.ip(), details.ip.port());
        }

        self.register_and_relay_node(details.clone()).await?;
        Ok(details)
    }

    /// Continuously listens for incoming node registrations and relays their IPs.
    pub async fn run(&mut self) -> Result<(), StruumError> {
        log::info!(
            "Relay server listening on UDP port {} / TCP port {}",
            self.net.udp_port,
            self.net.tcp_port
        );
        loop {
            if let Err(e) = self.listen_and_relay_once().await {
                log::error!("Error in relay listen loop: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use struum_types::network::{NodeBackend, NodeRole};

    #[tokio::test]
    async fn test_relay_server_creation_and_registration() {
        let mut relay = RelayServer::<1024>::new(39751, 39752)
            .await
            .expect("Failed to create RelayServer");

        let node1 = NodeDetails {
            id: UID::new(10),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38711),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
            tcp_port: Some(38712),
        };
        let node2 = NodeDetails {
            id: UID::new(20),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38721),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::OpenGL),
            tcp_port: Some(38722),
        };

        relay.register_and_relay_node(node1.clone()).await.unwrap();
        assert_eq!(relay.registered_count(), 1);

        relay.register_and_relay_node(node2.clone()).await.unwrap();
        assert_eq!(relay.registered_count(), 2);
        assert_eq!(relay.get_node(&UID::new(10)), Some(&node1));
        assert_eq!(relay.get_node(&UID::new(20)), Some(&node2));
    }
}
