//! Localhost tests for the transport: length-prefixed TCP framing and UDP datagrams.

use std::net::SocketAddr;
use std::time::Duration;

use struum_network::layer::{NetworkLayer, PeerConnection};
use struum_types::StruumError;
use struum_types::network::{
    IntroductionPacket, NodeBackend, NodeRole, ResultPacket, TcpPacket, UID, UdpPacket, WorkKind,
    WorkPacket,
};
use tokio::io::AsyncWriteExt;
use tokio::time::timeout;

type Layer = NetworkLayer<2048>;

const WAIT: Duration = Duration::from_secs(5);

async fn layer() -> Layer {
    // Port 0: let the OS pick, so tests can run in parallel.
    Layer::new(0, 0).await.expect("bind layer")
}

fn local(port: u16) -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], port))
}

fn work(data: Vec<u8>) -> TcpPacket {
    TcpPacket::WORK(WorkPacket {
        task_id: "job-1".into(),
        sender_id: UID::new(42),
        kind: WorkKind::Cpu,
        data,
    })
}

/// Connects `client` to `server` and returns (server's view of client, client's view of server).
async fn connect(server: &mut Layer, client: &mut Layer) -> (SocketAddr, SocketAddr) {
    let server_addr = local(server.tcp_port);
    let (accepted, connected) = tokio::join!(
        server.accept_tcp_connection(),
        client.initialize_tcp_connection_addr(server_addr)
    );
    connected.expect("connect");
    (accepted.expect("accept"), server_addr)
}

#[tokio::test]
async fn new_binds_os_assigned_ports() {
    let a = layer().await;
    let b = layer().await;
    assert_ne!(a.tcp_port, 0);
    assert_ne!(a.udp_port, 0);
    assert_ne!(a.tcp_port, b.tcp_port);
}

#[tokio::test]
async fn binding_a_taken_port_is_an_error() {
    let a = layer().await;
    let err = Layer::new(0, a.tcp_port).await.unwrap_err();
    assert!(matches!(err, StruumError::NetworkConnectionError(_)), "{err}");
}

#[tokio::test]
async fn tcp_packets_round_trip_both_ways() {
    let mut server = layer().await;
    let mut client = layer().await;
    let (client_addr, server_addr) = connect(&mut server, &mut client).await;

    client.send_tcp_data(server_addr, work(vec![1, 2, 3])).await.unwrap();
    assert_eq!(server.recv_tcp_data(&client_addr).await.unwrap(), work(vec![1, 2, 3]));

    let reply = TcpPacket::RESULT(ResultPacket::new("job-1".into(), UID::new(7), vec![9; 10]));
    server.send_tcp_data(client_addr, reply.clone()).await.unwrap();
    assert_eq!(client.recv_tcp_data(&server_addr).await.unwrap(), reply);
}

#[tokio::test]
async fn large_payloads_are_reassembled() {
    // Much larger than BUF_SIZE and a TCP segment: framing must read the whole payload.
    let mut server = layer().await;
    let mut client = layer().await;
    let (client_addr, server_addr) = connect(&mut server, &mut client).await;

    let data: Vec<u8> = (0..4 * 1024 * 1024).map(|i| (i % 251) as u8).collect();
    let packet = work(data);
    let sent = packet.clone();
    let send = async { client.send_tcp_data(server_addr, sent).await };
    let (sent_ok, received) = tokio::join!(send, server.recv_tcp_data(&client_addr));
    sent_ok.unwrap();
    assert_eq!(received.unwrap(), packet);
}

#[tokio::test]
async fn back_to_back_packets_keep_their_boundaries() {
    let mut server = layer().await;
    let mut client = layer().await;
    let (client_addr, server_addr) = connect(&mut server, &mut client).await;

    for i in 0..20u8 {
        client.send_tcp_data(server_addr, work(vec![i; i as usize])).await.unwrap();
    }
    for i in 0..20u8 {
        assert_eq!(server.recv_tcp_data(&client_addr).await.unwrap(), work(vec![i; i as usize]));
    }
}

#[tokio::test]
async fn peer_request_waits_for_the_reply() {
    let mut server = layer().await;
    let mut client = layer().await;
    let (client_addr, server_addr) = connect(&mut server, &mut client).await;

    let to_server = server.peer(&client_addr).unwrap();
    let echo = tokio::spawn(async move {
        let packet = to_server.recv().await.unwrap();
        to_server.send(&packet).await.unwrap();
    });

    let to_client = client.peer(&server_addr).unwrap();
    let reply = timeout(WAIT, to_client.request(&work(vec![5; 100]))).await.unwrap().unwrap();
    assert_eq!(reply, work(vec![5; 100]));
    echo.await.unwrap();
}

#[tokio::test]
async fn unknown_or_disconnected_peers_are_errors() {
    let mut server = layer().await;
    let mut client = layer().await;
    let nobody = local(1);
    assert!(client.send_tcp_data(nobody, work(vec![])).await.is_err());
    assert!(client.recv_tcp_data(&nobody).await.is_err());
    assert!(client.peer(&nobody).is_none());

    let (_, server_addr) = connect(&mut server, &mut client).await;
    assert!(client.peer(&server_addr).is_some());
    client.disconnect(&server_addr);
    assert!(client.peer(&server_addr).is_none());
    assert!(client.send_tcp_data(server_addr, work(vec![])).await.is_err());
}

#[tokio::test]
async fn closed_connection_is_reported_on_recv() {
    let mut server = layer().await;
    let mut client = layer().await;
    let (client_addr, server_addr) = connect(&mut server, &mut client).await;

    client.disconnect(&server_addr); // drops the only handle -> closes the socket
    let err = timeout(WAIT, server.recv_tcp_data(&client_addr)).await.unwrap().unwrap_err();
    assert!(matches!(err, StruumError::NetworkConnectionError(_)), "{err}");
}

#[tokio::test]
async fn garbage_frames_are_serialization_errors() {
    let mut server = layer().await;
    let server_addr = local(server.tcp_port);
    let (accepted, raw) =
        tokio::join!(server.accept_tcp_connection(), tokio::net::TcpStream::connect(server_addr));
    let client_addr = accepted.unwrap();
    let mut raw = raw.unwrap();

    // Valid length header, payload with an out-of-range enum tag.
    let payload = 99u32.to_le_bytes();
    raw.write_all(&(payload.len() as u32).to_be_bytes()).await.unwrap();
    raw.write_all(&payload).await.unwrap();

    let err = server.recv_tcp_data(&client_addr).await.unwrap_err();
    assert!(matches!(err, StruumError::SerializationError(_)), "{err}");
}

#[tokio::test]
async fn udp_introductions_are_delivered() {
    let mut a = layer().await;
    let mut b = layer().await;
    let intro = IntroductionPacket {
        id: UID::generate(),
        ip: local(a.udp_port),
        role: NodeRole::NODE,
        backend: Some(NodeBackend::Cpu),
        tcp_port: Some(a.tcp_port),
        quic_port: None,
    };

    a.send_udp_data(UdpPacket::INTRODUCTION(intro.clone()), local(b.udp_port))
        .await
        .unwrap();
    let (received, src) =
        timeout(WAIT, b.listen_udp::<IntroductionPacket>()).await.unwrap().unwrap();
    assert_eq!(received, intro);
    assert_eq!(src.port(), a.udp_port);
}

#[tokio::test]
async fn malformed_udp_datagrams_are_rejected() {
    let mut b = layer().await;
    let sender = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    sender.send_to(&[1, 2, 3], local(b.udp_port)).await.unwrap();

    let err = timeout(WAIT, b.listen_udp::<IntroductionPacket>()).await.unwrap().unwrap_err();
    assert!(matches!(err, StruumError::SerializationError(_)), "{err}");
}

#[test]
fn payload_frames_match_full_packet_frames() {
    let data: Vec<u8> = (0..1000).map(|i| (i % 7) as u8).collect();
    let frame = PeerConnection::frame_with_payload(&work(Vec::new()), data.len()).unwrap();
    let mut wire = frame.clone();
    wire.extend_from_slice(&data);

    let full = bincode::serialize(&work(data)).unwrap();
    assert_eq!(&wire[..4], &(full.len() as u32).to_be_bytes());
    assert_eq!(&wire[4..], full.as_slice());

    // A packet that doesn't end in an empty byte field is refused.
    assert!(PeerConnection::frame_with_payload(&work(vec![1]), 10).is_err());
}

#[tokio::test]
async fn request_with_payload_round_trips() {
    let mut server = layer().await;
    let mut client = layer().await;
    let (client_addr, server_addr) = connect(&mut server, &mut client).await;

    let to_server = server.peer(&client_addr).unwrap();
    let echo = tokio::spawn(async move {
        let packet = to_server.recv().await.unwrap();
        to_server.send(&packet).await.unwrap();
    });

    let data: Vec<u8> = (0..2 * 1024 * 1024).map(|i| (i % 253) as u8).collect();
    let reply = timeout(WAIT, client.peer(&server_addr).unwrap().request_with_payload(&work(Vec::new()), &data))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(reply, work(data));
    echo.await.unwrap();
}

#[tokio::test]
async fn oversized_length_headers_are_refused() {
    let mut server = layer().await;
    let server_addr = local(server.tcp_port);
    let (accepted, raw) =
        tokio::join!(server.accept_tcp_connection(), tokio::net::TcpStream::connect(server_addr));
    let client_addr = accepted.unwrap();
    let mut raw = raw.unwrap();

    raw.write_all(&u32::MAX.to_be_bytes()).await.unwrap();
    let err = timeout(WAIT, server.recv_tcp_data(&client_addr)).await.unwrap().unwrap_err();
    assert!(err.to_string().contains("limit"), "{err}");
}
