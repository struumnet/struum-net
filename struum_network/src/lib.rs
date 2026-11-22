//! Transport layer: UDP datagrams for discovery and length-prefixed TCP for work.
//!
//! [`NetworkLayer`](layer::NetworkLayer) owns one UDP socket and one TCP listener, plus a
//! pool of TCP connections keyed by peer address. Higher layers (the
//! [node](../struum_node/index.html) and the [relay](../struum_relay/index.html)) embed one.
//!
//! - **UDP**: bincode-encoded [`UdpPacket`](struum_types::network::UdpPacket) payloads,
//!   used for relay registration and introductions.
//! - **TCP**: each [`TcpPacket`](struum_types::network::TcpPacket) is sent as a 4-byte
//!   big-endian length followed by its bincode encoding.
//!   [`PeerConnection`](layer::PeerConnection) is a cloneable handle to one connection,
//!   for request/response exchanges from other tasks.
//! - **QUIC** ([`QuicLayer`](quic::QuicLayer)): one UDP socket that accepts and dials,
//!   with TLS 1.3 and a shared-token handshake. Each peer's packets travel on one
//!   stream with the same framing, behind the same [`PeerConnection`](layer::PeerConnection).

pub mod layer;
pub mod quic;
