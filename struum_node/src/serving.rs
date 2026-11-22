//! Accepts requesters and serves their work, each in its own task. Used by
//! `struum_worker` and `struum-tui`.
//!
//! QUIC is always on. With a token (`STRUUM_TOKEN`), only peers that prove
//! they know it get in, and plain TCP is off, because it has no
//! authentication and would bypass the token. Without a token, the node is
//! open and also serves TCP for requesters on older builds.
//!
//! [`start`] does everything at once. A node that registers with the relay
//! through its own socket (so the relay must learn the QUIC port and whether
//! TCP is on first) calls [`prepare`], registers, then [`spawn`].

use std::time::Duration;

use struum_types::StruumError;
use struum_types::network::NetworkCommunicator;

use crate::Node;

/// How often a serving node that registered with a relay
/// ([`Node::long_poll_relay`]) registers again, so the relay doesn't expire it
/// (`struum_relay::DEFAULT_TTL` is 30 s).
pub const HEARTBEAT: Duration = Duration::from_secs(10);

/// How requesters can reach this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Listening {
    pub quic_port: u16,
    /// `None` when TCP is off because a token is set.
    pub tcp_port: Option<u16>,
    /// Whether peers must prove they know the token.
    pub authenticated: bool,
}

/// Starts the QUIC endpoint (UDP `quic_port`, 0 picks one) authenticated by
/// `token`, and with a token, stops advertising TCP. After this,
/// [`Node::details`] is what requesters should be told.
pub fn prepare<const N: usize>(node: &mut Node<N>, quic_port: u16, token: &[u8]) -> Result<Listening, StruumError> {
    let quic_port = node.enable_quic(quic_port, token)?;
    let authenticated = !token.is_empty();
    node.set_advertise_tcp(!authenticated);
    Ok(Listening {
        quic_port,
        tcp_port: (!authenticated).then_some(node.net.tcp_port),
        authenticated,
    })
}

/// [`prepare`] then [`spawn`].
pub fn start<const N: usize>(mut node: Node<N>, quic_port: u16, token: &[u8]) -> Result<Listening, StruumError> {
    let listening = prepare(&mut node, quic_port, token)?;
    spawn(node, listening);
    Ok(listening)
}

/// Serves requesters on a node set up by [`prepare`]: every QUIC peer, and
/// TCP ones if `listening.tcp_port` is set. Takes the node: from here on it
/// only serves. If the node registered with a relay, it re-registers every
/// [`HEARTBEAT`].
///
/// # Panics
/// If QUIC isn't enabled on `node`.
pub fn spawn<const N: usize>(node: Node<N>, listening: Listening) {
    spawn_with_heartbeat(node, listening, HEARTBEAT);
}

/// [`spawn`] re-registering with the relay every `heartbeat`.
pub fn spawn_with_heartbeat<const N: usize>(node: Node<N>, listening: Listening, heartbeat: Duration) {
    let quic = node.quic().expect("serving::spawn needs QUIC enabled (serving::prepare)");
    let server = node.server();

    tokio::spawn(async move {
        loop {
            match quic.accept().await {
                Ok(peer) => {
                    log::info!("Serving requester {} over QUIC from {}", peer.id, peer.remote_addr());
                    let server = server.clone();
                    tokio::spawn(async move {
                        server.serve(peer.conn.clone()).await;
                        peer.close();
                    });
                }
                Err(e) => {
                    log::warn!("QUIC endpoint stopped: {e}");
                    return;
                }
            }
        }
    });

    // This task also keeps the node (its TCP listener and scheduler) alive,
    // and keeps it registered with the relay it registered with.
    tokio::spawn(async move {
        let mut node = node;
        let tcp_on = listening.tcp_port.is_some();
        if !tcp_on {
            log::info!("TCP is off: requesters must use QUIC");
        }
        let relay = node.relay_addr;
        let mut beat = tokio::time::interval(heartbeat);
        beat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                res = node.accept_peer(), if tcp_on => match res {
                    Ok(peer) => {
                        log::info!("Serving requester {} over TCP", peer.addr);
                        tokio::spawn(node.server().serve(peer));
                    }
                    Err(e) => {
                        log::warn!("Accept failed: {e}");
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                },
                _ = beat.tick(), if relay.is_some() => {
                    let relay = relay.expect("checked by the guard");
                    if let Err(e) = node.introduce(relay).await {
                        log::debug!("Relay heartbeat to {relay} failed: {e}");
                    }
                }
                else => std::future::pending::<()>().await,
            }
        }
    });
}
