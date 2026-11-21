//! Keeps this node registered with the relay and collects peer introductions.
//!
//! The [`Node`](struum_node::Node) is busy serving requesters (which needs
//! `&mut Node`), so discovery runs on a separate UDP socket. It registers with
//! the relay under the node's id and TCP port, so requesters introduced to us
//! connect straight to the serving node, while introductions come back to this
//! socket.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use struum_types::network::{IntroductionPacket, NodeBackend, NodeDetails, NodeRole, UID};
use tokio::net::UdpSocket;

/// How the relay connection looks from here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelayStatus {
    /// Registration sent, no introduction received yet (the relay only answers
    /// once another node is registered).
    Registering,
    /// The relay has introduced at least one peer.
    Connected,
    /// Sending to the relay failed.
    Error(String),
}

/// A peer the relay introduced.
#[derive(Debug, Clone)]
pub struct KnownPeer {
    pub details: NodeDetails,
    pub first_seen: Instant,
    pub last_seen: Instant,
}

/// Shared state written by the discovery task and read by the UI.
#[derive(Debug)]
pub struct Discovery {
    pub relay: SocketAddr,
    pub status: RelayStatus,
    pub peers: HashMap<UID, KnownPeer>,
    pub last_registration: Option<Instant>,
}

impl Discovery {
    pub fn new(relay: SocketAddr) -> Self {
        Self { relay, status: RelayStatus::Registering, peers: HashMap::new(), last_registration: None }
    }

    /// Records an introduction (ignoring ones about ourselves).
    pub fn introduce(&mut self, me: UID, details: NodeDetails) {
        if details.id == me {
            return;
        }
        let now = Instant::now();
        self.peers
            .entry(details.id)
            .and_modify(|p| {
                p.details = details.clone();
                p.last_seen = now;
            })
            .or_insert(KnownPeer { details, first_seen: now, last_seen: now });
        self.status = RelayStatus::Connected;
    }
}

pub type SharedDiscovery = Arc<Mutex<Discovery>>;

/// What we tell the relay about ourselves.
#[derive(Debug, Clone, Copy)]
pub struct Registration {
    pub id: UID,
    /// Address to advertise, normally the node's LAN IP (as `Node` itself does).
    /// Unspecified lets the relay use the address it sees us from.
    pub ip: IpAddr,
    pub backend: NodeBackend,
    pub tcp_port: u16,
}

/// Binds the discovery socket. Returns it with its local UDP port.
pub async fn bind(port: u16) -> std::io::Result<(UdpSocket, u16)> {
    let socket = UdpSocket::bind(("0.0.0.0", port)).await?;
    let port = socket.local_addr()?.port();
    Ok((socket, port))
}

/// Registers every `interval` and records introductions forever.
pub async fn run(socket: UdpSocket, me: Registration, shared: SharedDiscovery, interval: Duration) {
    let relay = shared.lock().unwrap().relay;
    let port = socket.local_addr().map(|a| a.port()).unwrap_or(0);
    let intro = IntroductionPacket {
        id: me.id,
        ip: SocketAddr::new(me.ip, port),
        role: NodeRole::NODE,
        backend: Some(me.backend),
        tcp_port: Some(me.tcp_port),
    };
    let packet = bincode::serialize(&intro).expect("introduction serializes");
    let mut buf = vec![0u8; 2048];
    let mut next = tokio::time::Instant::now();
    loop {
        if tokio::time::Instant::now() >= next {
            let sent = socket.send_to(&packet, relay).await;
            let mut d = shared.lock().unwrap();
            d.last_registration = Some(Instant::now());
            match sent {
                Ok(_) if matches!(d.status, RelayStatus::Error(_)) => d.status = RelayStatus::Registering,
                Ok(_) => {}
                Err(e) => {
                    log::warn!("Relay {relay} unreachable: {e}");
                    d.status = RelayStatus::Error(e.to_string());
                }
            }
            next = tokio::time::Instant::now() + interval;
        }
        tokio::select! {
            res = socket.recv_from(&mut buf) => match res {
                Ok((n, _src)) => match bincode::deserialize::<IntroductionPacket>(&buf[..n]) {
                    Ok(intro) => {
                        let details = NodeDetails::from(intro);
                        let mut d = shared.lock().unwrap();
                        if details.id != me.id && !d.peers.contains_key(&details.id) {
                            log::info!(
                                "Relay introduced node {} at {} ({:?})",
                                details.id, details.ip, details.backend
                            );
                        }
                        d.introduce(me.id, details);
                    }
                    Err(e) => log::debug!("Ignoring malformed discovery packet: {e}"),
                },
                Err(e) => log::debug!("Discovery socket error: {e}"),
            },
            _ = tokio::time::sleep_until(next) => {}
        }
    }
}
