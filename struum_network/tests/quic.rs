//! QUIC transport on localhost: authentication and packet exchange.

use std::net::SocketAddr;
use std::time::Duration;

use struum_network::quic::QuicLayer;
use struum_types::network::{TcpPacket, UID, WorkKind, WorkPacket};

fn local() -> SocketAddr {
    "127.0.0.1:0".parse().unwrap()
}

fn work(task_id: &str, data: Vec<u8>) -> TcpPacket {
    TcpPacket::WORK(WorkPacket { task_id: task_id.into(), sender_id: UID::new(1), kind: WorkKind::Cpu, data })
}

/// A server that echoes every packet back on the same connection.
fn spawn_echo(server: QuicLayer) {
    tokio::spawn(async move {
        while let Ok(peer) = server.accept().await {
            tokio::spawn(async move {
                while let Ok(packet) = peer.conn.recv().await {
                    if peer.conn.send(&packet).await.is_err() {
                        return;
                    }
                }
            });
        }
    });
}

#[tokio::test]
async fn peers_with_the_same_token_exchange_packets() {
    let server = QuicLayer::bind(local(), 20, b"secret").unwrap();
    let addr = server.local_addr().unwrap();
    spawn_echo(server);

    let client = QuicLayer::bind(local(), 10, b"secret").unwrap();
    let peer = client.connect(addr).await.expect("authenticated connection");
    assert_eq!(peer.id, 20);

    // Several requests, a large one among them, come back in order.
    let big = vec![7u8; 8 << 20];
    for (i, data) in [vec![1, 2, 3], big.clone(), Vec::new()].into_iter().enumerate() {
        let reply = peer.conn.request(&work(&format!("t{i}"), data.clone())).await.unwrap();
        match reply {
            TcpPacket::WORK(w) => {
                assert_eq!(w.task_id, format!("t{i}"));
                assert_eq!(w.data, data);
            }
            other => panic!("unexpected reply {other:?}"),
        }
    }
}

#[tokio::test]
async fn pipelined_payload_sends_match_replies_in_order() {
    let server = QuicLayer::bind(local(), 21, b"").unwrap();
    let addr = server.local_addr().unwrap();
    spawn_echo(server);
    let client = QuicLayer::bind(local(), 11, b"").unwrap();
    let peer = client.connect(addr).await.unwrap();

    // Hold the reader, send several frames back to back, then read them all.
    let mut reader = peer.conn.reader().await;
    for i in 0..4u8 {
        let header = work(&format!("p{i}"), Vec::new());
        peer.conn.send_with_payload(&header, &vec![i; 1 << 20]).await.unwrap();
    }
    for i in 0..4u8 {
        match reader.recv().await.unwrap() {
            TcpPacket::WORK(w) => {
                assert_eq!(w.task_id, format!("p{i}"));
                assert!(w.data.len() == 1 << 20 && w.data.iter().all(|b| *b == i));
            }
            other => panic!("unexpected reply {other:?}"),
        }
    }
}

#[tokio::test]
async fn a_wrong_token_is_rejected_on_both_sides() {
    let server = QuicLayer::bind(local(), 22, b"right").unwrap();
    let addr = server.local_addr().unwrap();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    tokio::spawn(async move {
        // accept() skips peers that fail to authenticate; report any that get through.
        // Keep them (and the endpoint) open: dropping the last handle closes the connection.
        let mut kept = Vec::new();
        while let Ok(peer) = server.accept().await {
            let _ = tx.send(peer.id);
            kept.push(peer);
        }
    });

    let client = QuicLayer::bind(local(), 12, b"wrong").unwrap();
    let err = client.connect(addr).await.unwrap_err().to_string();
    assert!(!err.is_empty());
    let accepted = tokio::time::timeout(Duration::from_millis(500), rx.recv()).await;
    assert!(accepted.is_err(), "server must not accept a peer with the wrong token: {accepted:?}");

    // A peer with the right token still gets in afterwards.
    let good = QuicLayer::bind(local(), 13, b"right").unwrap();
    good.connect(addr).await.expect("right token");
    assert_eq!(rx.recv().await, Some(13));
}

#[tokio::test]
async fn a_closed_peer_fails_reads_instead_of_hanging() {
    let server = QuicLayer::bind(local(), 23, b"").unwrap();
    let addr = server.local_addr().unwrap();
    let client = QuicLayer::bind(local(), 14, b"").unwrap();
    let (accepted, connected) = tokio::join!(server.accept(), client.connect(addr));
    let (theirs, ours) = (accepted.unwrap(), connected.unwrap());
    theirs.close();
    let read = tokio::time::timeout(Duration::from_secs(5), ours.conn.recv()).await;
    assert!(matches!(read, Ok(Err(_))), "{read:?}");
    assert!(ours.conn.is_broken());
}
