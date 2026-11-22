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

use crate::Node;

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
/// only serves.
///
/// # Panics
/// If QUIC isn't enabled on `node`.
pub fn spawn<const N: usize>(node: Node<N>, listening: Listening) {
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

    // This task also keeps the node (its TCP listener and scheduler) alive.
    tokio::spawn(async move {
        if listening.tcp_port.is_none() {
            log::info!("TCP is off: requesters must use QUIC");
            std::future::pending::<()>().await;
        }
        loop {
            match node.accept_peer().await {
                Ok(peer) => {
                    log::info!("Serving requester {} over TCP", peer.addr);
                    tokio::spawn(node.server().serve(peer));
                }
                Err(e) => {
                    log::warn!("Accept failed: {e}");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    });
}
