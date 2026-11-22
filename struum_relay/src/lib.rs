//! Relay server for peer discovery.
//!
//! Nodes send an [`IntroductionPacket`] to the relay over UDP. The relay records the node in
//! its registry and introduces it to every node already registered, in both directions,
//! so the nodes can then connect to each other. If a node reports an
//! unspecified IP (`0.0.0.0`), the relay uses the packet's source address.
//!
//! Registrations expire: a node that hasn't re-registered within the relay's TTL
//! ([`DEFAULT_TTL`], `RELAY_TTL_SECS` for the binary) is dropped and no longer handed
//! out. Re-registering with unchanged details is a refresh: the node is sent the
//! current registry again (in case UDP lost something), but the others aren't told,
//! since they already know it. Only new or changed nodes are announced to everyone.
//!
//! Run it with `cargo run -p struum_relay`, or embed a [`RelayServer`]:
//!
//! ```no_run
//! # async fn run() -> Result<(), struum_types::StruumError> {
//! let mut relay = struum_relay::RelayServer::<1024>::new(39001, 39002).await?;
//! relay.run().await
//! # }
//! ```

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use struum_network::layer::NetworkLayer;
use struum_types::StruumError;
use struum_types::network::{IntroductionPacket, NodeDetails, UID, UdpPacket};

/// How long a registration lasts without a refresh. Nodes re-register well
/// within this (`struum-tui` every 2 s, serving nodes every
/// `struum_node::serving::HEARTBEAT`).
pub const DEFAULT_TTL: Duration = Duration::from_secs(30);

/// `RELAY_TTL_SECS`, or [`DEFAULT_TTL`].
pub fn ttl_from_env() -> Duration {
    std::env::var("RELAY_TTL_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TTL)
}

fn intro(details: &NodeDetails) -> UdpPacket {
    UdpPacket::INTRODUCTION(IntroductionPacket {
        id: details.id,
        ip: details.ip,
        role: details.role,
        backend: details.backend,
        tcp_port: details.tcp_port,
        quic_port: details.quic_port,
    })
}

/// RelayServer coordinates and relays node IP registrations.
/// Nodes register their IP and role by sending an IntroductionPacket to the relay.
/// The relay records the node and relays the newly registered IP to all existing nodes,
/// as well as relaying all existing nodes' IPs to the new node.
pub struct RelayServer<const BUF_SIZE: usize> {
    /// The relay's own identifier.
    pub id: UID,
    /// Sockets the relay listens and sends on.
    pub net: NetworkLayer<BUF_SIZE>,
    /// Every live registration, by id.
    pub registered_nodes: HashMap<UID, NodeDetails>,
    /// Registrations not refreshed for this long are dropped.
    pub ttl: Duration,
    /// When each registered node last registered.
    last_seen: HashMap<UID, Instant>,
}

impl<const BUF_SIZE: usize> RelayServer<BUF_SIZE> {
    /// Creates a new RelayServer listening on the specified UDP and TCP ports,
    /// with registrations lasting [`DEFAULT_TTL`].
    pub async fn new(udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
        let net = NetworkLayer::new(udp_port, tcp_port).await?;
        Ok(Self {
            id: UID::generate(),
            net,
            registered_nodes: HashMap::default(),
            ttl: DEFAULT_TTL,
            last_seen: HashMap::default(),
        })
    }

    /// Sets how long registrations last without a refresh.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Number of registered nodes currently stored in the relay.
    pub fn registered_count(&self) -> usize {
        self.registered_nodes.len()
    }

    /// Returns a registered node by UID.
    pub fn get_node(&self, id: &UID) -> Option<&NodeDetails> {
        self.registered_nodes.get(id)
    }

    /// Drops registrations older than the TTL and returns their ids.
    pub fn expire_stale(&mut self) -> Vec<UID> {
        let ttl = self.ttl;
        let stale: Vec<UID> = self
            .last_seen
            .iter()
            .filter(|(_, seen)| seen.elapsed() > ttl)
            .map(|(id, _)| *id)
            .collect();
        for id in &stale {
            self.last_seen.remove(id);
            if let Some(node) = self.registered_nodes.remove(id) {
                log::info!("Relay dropped node {} at {}: no registration for {:?}", id, node.ip, ttl);
            }
        }
        stale
    }

    /// Registers (or refreshes) a node and sends it every other live node.
    /// Other nodes are only told about it when it is new or its details changed.
    /// A failed send is logged and skipped, so one unreachable node can't stop
    /// the others from being introduced.
    pub async fn register_and_relay_node(&mut self, details: NodeDetails) -> Result<(), StruumError> {
        self.expire_stale();
        let announce = self.registered_nodes.get(&details.id) != Some(&details);
        if !self.registered_nodes.contains_key(&details.id) {
            log::info!(
                "Relay registered node {} at IP {} (backend: {:?})",
                details.id,
                details.ip,
                details.backend
            );
        } else if announce {
            log::info!("Relay updated node {} at IP {}", details.id, details.ip);
        } else {
            log::debug!("Relay refreshed registration for node {} at IP {}", details.id, details.ip);
        }

        let others: Vec<NodeDetails> =
            self.registered_nodes.values().filter(|n| n.id != details.id).cloned().collect();
        for other in &others {
            if announce {
                log::info!("Relaying IPs: introducing {} ({}) <---> {} ({})", details.id, details.ip, other.id, other.ip);
                if let Err(e) = self.net.send_udp_data(intro(&details), other.ip).await {
                    log::warn!("Relay couldn't introduce {} to {} at {}: {e}", details.id, other.id, other.ip);
                }
            }
            if let Err(e) = self.net.send_udp_data(intro(other), details.ip).await {
                log::warn!("Relay couldn't introduce {} to {} at {}: {e}", other.id, details.id, details.ip);
            }
        }

        self.last_seen.insert(details.id, Instant::now());
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
            "Relay server listening on UDP port {} / TCP port {} (registrations last {:?})",
            self.net.udp_port,
            self.net.tcp_port,
            self.ttl
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
        let mut relay = RelayServer::<1024>::new(0, 0)
            .await
            .expect("Failed to create RelayServer");

        let node1 = NodeDetails {
            id: UID::new(10),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38711),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
            tcp_port: Some(38712),
            quic_port: None,
        };
        let node2 = NodeDetails {
            id: UID::new(20),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38721),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::OpenGL),
            tcp_port: Some(38722),
            quic_port: None,
        };

        relay.register_and_relay_node(node1.clone()).await.unwrap();
        assert_eq!(relay.registered_count(), 1);

        relay.register_and_relay_node(node2.clone()).await.unwrap();
        assert_eq!(relay.registered_count(), 2);
        assert_eq!(relay.get_node(&UID::new(10)), Some(&node1));
        assert_eq!(relay.get_node(&UID::new(20)), Some(&node2));
    }
}
