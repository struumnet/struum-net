//! Accepts requesters and serves their work, each in its own task.
//!
//! QUIC is always on. With a token (`STRUUM_TOKEN`), only peers that prove
//! they know it get in, and plain TCP is off, because it has no
//! authentication and would bypass the token. Without a token, the node is
//! open and also serves TCP for requesters on older builds.

use std::time::Duration;

use struum_node::Node;
use struum_types::StruumError;

/// How requesters can reach this node: shown in the header and sent to the relay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Listening {
    pub quic_port: u16,
    /// `None` when TCP is off because a token is set.
    pub tcp_port: Option<u16>,
    /// Whether peers must prove they know the token.
    pub authenticated: bool,
}

/// Starts serving on the QUIC endpoint (UDP `quic_port`, 0 picks one) and,
/// without a token, on the node's TCP listener. Takes the node: from here on
/// it only serves.
pub fn start<const N: usize>(mut node: Node<N>, quic_port: u16, token: &[u8]) -> Result<Listening, StruumError> {
    let quic_port = node.enable_quic(quic_port, token)?;
    let quic = node.quic().expect("just enabled");
    let server = node.server();
    let authenticated = !token.is_empty();
    let listening = Listening {
        quic_port,
        tcp_port: (!authenticated).then_some(node.net.tcp_port),
        authenticated,
    };

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
        if authenticated {
            log::info!("Token set: TCP is off, requesters must use QUIC");
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
    Ok(listening)
}
