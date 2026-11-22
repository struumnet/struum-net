//! Wire-format and buffer edge cases. Packets are bincode-encoded between
//! devices that may run different builds, so these pin down the encoding.

use std::net::SocketAddr;

use struum_types::network::{
    IntroductionPacket, NodeBackend, NodeDetails, NodeRole, ResultPacket, TcpPacket, UID,
    WorkErrorPacket, WorkKind, WorkPacket,
};
use struum_types::{Buffer, StruumError};

fn addr(s: &str) -> SocketAddr {
    s.parse().unwrap()
}

#[test]
fn uid_encodes_as_eight_bytes_and_keeps_full_width() {
    let uid = UID::new(u64::MAX - 1);
    let bytes = bincode::serialize(&uid).unwrap();
    assert_eq!(bytes.len(), 8);
    assert_eq!(bincode::deserialize::<UID>(&bytes).unwrap(), uid);
    assert_eq!(u64::from(uid), u64::MAX - 1);
    assert_eq!(*UID::from(7u64), 7);
}

#[test]
fn introduction_round_trips_into_node_details() {
    let intro = IntroductionPacket {
        id: UID::generate(),
        ip: addr("192.168.1.20:4000"),
        role: NodeRole::NODE,
        backend: Some(NodeBackend::OpenGL),
        tcp_port: Some(4001),
        quic_port: None,
    };
    let decoded: IntroductionPacket =
        bincode::deserialize(&bincode::serialize(&intro).unwrap()).unwrap();
    assert_eq!(decoded, intro);

    let details = NodeDetails::from(decoded);
    assert_eq!(details.id, intro.id);
    assert_eq!(details.tcp_addr(), Some(addr("192.168.1.20:4001")));
}

#[test]
fn node_without_tcp_port_has_no_tcp_addr() {
    let details = NodeDetails {
        id: UID::new(1),
        ip: addr("10.0.0.1:5000"),
        role: NodeRole::COORDINATOR,
        backend: None,
        tcp_port: None,
        quic_port: None,
    };
    assert_eq!(details.tcp_addr(), None);
}

#[test]
fn tcp_packet_variant_tags_are_stable() {
    // New variants must be appended; peers on older builds rely on these tags.
    let work = TcpPacket::WORK(WorkPacket {
        task_id: "t".into(),
        sender_id: UID::new(1),
        kind: WorkKind::Cpu,
        data: vec![1, 2, 3],
    });
    let err = TcpPacket::WORKERROR(WorkErrorPacket {
        task_id: "t".into(),
        sender_id: UID::new(1),
        message: "boom".into(),
    });
    let result = TcpPacket::RESULT(ResultPacket::new("j".into(), UID::new(2), vec![9]));
    for (packet, tag) in [(&result, 3u32), (&work, 4), (&err, 5)] {
        let bytes = bincode::serialize(packet).unwrap();
        assert_eq!(u32::from_le_bytes(bytes[..4].try_into().unwrap()), tag, "{packet:?}");
        assert_eq!(&bincode::deserialize::<TcpPacket>(&bytes).unwrap(), packet);
    }
}

#[test]
fn truncated_packet_fails_to_decode() {
    let packet = TcpPacket::WORK(WorkPacket {
        task_id: "task".into(),
        sender_id: UID::new(3),
        kind: WorkKind::Gpu,
        data: vec![0; 64],
    });
    let bytes = bincode::serialize(&packet).unwrap();
    assert!(bincode::deserialize::<TcpPacket>(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn buffer_from_rows_validates_row_length() {
    let rows = Buffer::from_rows("img", &[0f32; 12], 4).unwrap();
    assert_eq!(rows.actual_len(), 3);
    assert_eq!(rows.slice(1..2).unwrap().as_slice::<f32>().len(), 4);

    assert!(matches!(
        Buffer::from_rows("img", &[0f32; 10], 4),
        Err(StruumError::ShardError(_))
    ));
    assert!(Buffer::from_rows("img", &[0f32; 4], 0).is_err());
}

#[test]
fn buffer_edge_cases() {
    let empty = Buffer::empty::<u32>("e", 0);
    assert_eq!(empty.actual_len(), 0);
    assert_eq!(empty.slice(0..0).unwrap().actual_len(), 0);

    let zero_stride = Buffer { name: "z".into(), data: vec![1, 2], stride: 0 };
    assert_eq!(zero_stride.actual_len(), 0);

    let buf = Buffer::new("x", &[1u32, 2, 3]);
    #[allow(clippy::reversed_empty_ranges)]
    let backwards = buf.slice(2..1);
    assert!(backwards.is_err());

    assert!(Buffer::concat(&[]).is_err());
    let other_name = Buffer::new("y", &[1u32]);
    let other_stride = Buffer::new("x", &[1u64]);
    assert!(Buffer::concat(&[buf.clone(), other_name]).is_err());
    assert!(Buffer::concat(&[buf, other_stride]).is_err());
}

#[test]
fn errors_display_their_kind_and_message() {
    let e = StruumError::NetworkConnectionError("refused".into());
    assert_eq!(e.to_string(), "Network Connection Error: refused");
    let boxed: Box<dyn std::error::Error> = Box::new(StruumError::ShardError("x".into()));
    assert_eq!(boxed.to_string(), "Shard Error: x");
}

/// What `Buffer` looked like before `serde_bytes`: bytes encoded one by one.
#[derive(serde_derive::Serialize)]
struct PlainBuffer {
    name: String,
    data: Vec<u8>,
    stride: usize,
}

#[test]
fn byte_fields_keep_the_same_encoding() {
    let data: Vec<u8> = (0..=255).collect();
    let buf = Buffer { name: "x".into(), data: data.clone(), stride: 4 };
    let plain = PlainBuffer { name: "x".into(), data, stride: 4 };
    assert_eq!(bincode::serialize(&buf).unwrap(), bincode::serialize(&plain).unwrap());
}
