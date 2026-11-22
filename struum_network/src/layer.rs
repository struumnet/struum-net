use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::net::{IpAddr, SocketAddr};
use std::net::AddrParseError;
use std::sync::Arc;
use struum_types::StruumError::NetworkConnectionError;
use struum_types::{
    StruumError,
    network::{TcpPacket, UdpPacket},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::{Mutex, MutexGuard};

/// Network layer is an Abstract layers that allows for any higher-level objects
/// like `Node`, `Coordinator` to consist of a Network Layer for itself.
#[derive(Debug)]
pub struct NetworkLayer<const BUF_SIZE: usize> {
    /// Public(Local) IPv4 address of itself, fetched during an outbound UDP connection.
    pub ip: IpAddr,
    /// A pool of connections of SocketAddresses with live connection made with other `NetworkLayer`s.
    connections: HashMap<SocketAddr, Connection>,
    /// A reserved port to be used by the OS for UDP communication.
    pub udp_port: u16,
    /// A reserved port to be used by the OS for TCP communication.
    pub tcp_port: u16,
    /// Persistent bound UDP socket used for both receiving and sending.
    pub udp_socket: Arc<UdpSocket>,
    /// TCP listener bound to tcp_port for accepting incoming connections.
    pub tcp_listener: TcpListener,
}

impl<const BUF_SIZE: usize> NetworkLayer<BUF_SIZE> {
    pub async fn new(udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
        let bind_addr = format!("0.0.0.0:{}", udp_port);
        let udp_socket = UdpSocket::bind(&bind_addr)
            .await
            .map_err(|e| StruumError::NetworkConnectionError(format!("Failed to bind UDP port {}: {}", udp_port, e)))?;
        let _ = udp_socket.set_broadcast(true);
        let actual_udp_port = udp_socket.local_addr().map(|a| a.port()).unwrap_or(udp_port);

        let tcp_bind_addr = format!("0.0.0.0:{}", tcp_port);
        let tcp_listener = TcpListener::bind(&tcp_bind_addr)
            .await
            .map_err(|e| StruumError::NetworkConnectionError(format!("Failed to bind TCP port {}: {}", tcp_port, e)))?;
        let actual_tcp_port = tcp_listener.local_addr().map(|a| a.port()).unwrap_or(tcp_port);

        Ok(Self {
            ip: get_local_ip().await?,
            connections: HashMap::default(),
            udp_port: actual_udp_port,
            tcp_port: actual_tcp_port,
            udp_socket: Arc::new(udp_socket),
            tcp_listener,
        })
    }
}

/// One live TCP connection plus what we know about the peer on the other end.
#[derive(Debug, Clone)]
struct Connection {
    reader: Arc<Mutex<OwnedReadHalf>>,
    writer: Arc<Mutex<OwnedWriteHalf>>,
    known_blobs: Arc<std::sync::Mutex<HashSet<u64>>>,
    broken: Arc<AtomicBool>,
}

impl Connection {
    fn new(stream: TcpStream) -> Self {
        let (reader, writer) = stream.into_split();
        Self {
            reader: Arc::new(Mutex::new(reader)),
            writer: Arc::new(Mutex::new(writer)),
            known_blobs: Arc::default(),
            broken: Arc::default(),
        }
    }
}

/// Socket buffer size for peer connections: large shards on a high-latency
/// Wi-Fi link need more than the default window in flight.
const SOCKET_BUFFER_BYTES: usize = 4 << 20;

/// Largest packet accepted from a peer. The length header comes off the wire, so
/// without a limit a corrupt or hostile peer could make us allocate up to 4 GiB.
pub const MAX_FRAME_BYTES: usize = 1 << 30;

/// Request/response tuning applied to every peer connection.
fn tune_stream(stream: &TcpStream) {
    // Don't let Nagle + delayed ACK stall small packets.
    let _ = stream.set_nodelay(true);
    let sock = socket2::SockRef::from(stream);
    let _ = sock.set_send_buffer_size(SOCKET_BUFFER_BYTES);
    let _ = sock.set_recv_buffer_size(SOCKET_BUFFER_BYTES);
}

/// Gives a local IP address for the NetworkLayer
async fn get_local_ip() -> Result<IpAddr, StruumError> {
    let socket = UdpSocket::bind("0.0.0.0:0")
        .await
        .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;

    // Pseudo connection request required to be made to resolve an IP
    socket
        .connect("8.8.8.8:80")
        .await
        .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;

    Ok(socket
        .local_addr()
        .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?
        .ip())
}

impl<const BUF_SIZE: usize> NetworkLayer<BUF_SIZE> {
    /// This methods allows for the nodes to initialize a targeted tcp connection within their own
    /// Connects to a target TCP address and stores the stream in connections.
    pub async fn initialize_tcp_connection_addr(
        &mut self,
        target_addr: SocketAddr,
    ) -> Result<(), StruumError> {
        let stream = TcpStream::connect(target_addr)
            .await
            .map_err(|e| NetworkConnectionError(format!("Failed to connect to {}: {}", target_addr, e)))?;
        tune_stream(&stream);

        self.connections
            .insert(target_addr, Connection::new(stream));

        Ok(())
    }

    /// This methods allows for the nodes to initialize a targeted tcp connection within their own
    /// `NetworkLayer<T>`. To communicate directly with the node.
    pub async fn initialize_tcp_connection(
        &mut self,
        target_ip: &str,
        target_port: u16,
    ) -> Result<(), StruumError> {
        let target_addr: SocketAddr = format!("{}:{}", target_ip, target_port)
            .parse()
            .map_err(|e: AddrParseError| StruumError::NetworkConnectionError(e.to_string()))?;

        self.initialize_tcp_connection_addr(target_addr).await
    }

    /// Broadcasts a UDP packet to the local devices
    pub async fn broadcast(&mut self, packet: UdpPacket) -> Result<(), StruumError> {
        let mut buf: Vec<u8>;
        match packet {
            UdpPacket::HELLO(data) => {
                buf = bincode::serialize(&data).map_err(|_| {
                    StruumError::NetworkConnectionError("Failed to serialize packet!".to_string())
                })?;
            }
            UdpPacket::INTRODUCTION(data) => {
                buf = bincode::serialize(&data).map_err(|_| {
                    StruumError::NetworkConnectionError("Failed to serialize packet!".to_string())
                })?;
            }
        }

        self.udp_socket
            .send_to(&mut buf, format!("255.255.255.255:{}", self.udp_port))
            .await
            .map_err(|e| {
                StruumError::NetworkConnectionError(format!("Failed to broadcast packet: {}", e))
            })?;

        Ok(())
    }

    pub async fn send_udp_data(
        &mut self,
        packet: UdpPacket,
        addr: SocketAddr,
    ) -> Result<(), StruumError> {
        let mut buf: Vec<u8>;
        match packet {
            UdpPacket::HELLO(data) => {
                buf = bincode::serialize(&data).map_err(|_| {
                    StruumError::NetworkConnectionError("Failed to serialize packet!".to_string())
                })?;
            }
            UdpPacket::INTRODUCTION(data) => {
                buf = bincode::serialize(&data).map_err(|_| {
                    StruumError::NetworkConnectionError("Failed to serialize packet!".to_string())
                })?;
            }
        }

        self.udp_socket
            .send_to(&mut buf, addr)
            .await
            .map_err(|e| {
                StruumError::NetworkConnectionError(format!("Failed to send UDP packet to {}: {}", addr, e))
            })?;

        Ok(())
    }

    /// This method is used for node to listen for any UDP Packets
    pub async fn listen_udp<T: Debug + for<'a> Deserialize<'a>>(
        &mut self,
    ) -> Result<(T, SocketAddr), StruumError> {
        let mut buf = [0; BUF_SIZE];

        let (amt, src) = self.udp_socket
            .recv_from(&mut buf)
            .await
            .map_err(|e| {
                StruumError::NetworkConnectionError(format!("Failed to receive UDP packet: {}", e))
            })?;
        let message = bincode::deserialize::<T>(&buf[..amt])
            .map_err(|e| StruumError::SerializationError(e.to_string()))?;
        log::debug!("Received {:?} from {:?}", message, src);
        Ok((message, src))
    }

    /// Sends a packet over the connection to `addr` established with
    /// `initialize_tcp_connection*()` or `accept_tcp_connection()`.
    /// Uses length-prefixed framing: 4-byte big-endian length header followed by bincode payload.
    pub async fn send_tcp_data(
        &mut self,
        addr: SocketAddr,
        data: TcpPacket,
    ) -> Result<(), StruumError> {
        self.peer(&addr)
            .ok_or(NetworkConnectionError(
                "Failed to find existing connection between nodes!".to_string(),
            ))?
            .send(&data)
            .await
    }

    /// Returns a cloneable handle to the connection with `addr`, usable from
    /// other tasks without borrowing the `NetworkLayer`.
    pub fn peer(&self, addr: &SocketAddr) -> Option<PeerConnection> {
        self.connections.get(addr).map(|conn| PeerConnection {
            addr: *addr,
            reader: conn.reader.clone(),
            writer: conn.writer.clone(),
            known_blobs: conn.known_blobs.clone(),
            broken: conn.broken.clone(),
        })
    }

    /// Drops the connection with `addr`, if any.
    pub fn disconnect(&mut self, addr: &SocketAddr) {
        self.connections.remove(addr);
    }

    /// Accepts an incoming TCP connection, stores it in the connection pool,
    /// and returns the peer's SocketAddr.
    pub async fn accept_tcp_connection(&mut self) -> Result<SocketAddr, StruumError> {
        let (stream, peer_addr) = self
            .tcp_listener
            .accept()
            .await
            .map_err(|e| NetworkConnectionError(format!("Failed to accept TCP connection: {}", e)))?;
        tune_stream(&stream);
        log::info!("Accepted TCP connection from {}", peer_addr);
        self.connections
            .insert(peer_addr, Connection::new(stream));
        Ok(peer_addr)
    }

    /// Reads a length-prefixed TcpPacket from an established connection.
    pub async fn recv_tcp_data(
        &mut self,
        addr: &SocketAddr,
    ) -> Result<TcpPacket, StruumError> {
        self.peer(addr)
            .ok_or(NetworkConnectionError(format!("No TCP connection found for {}", addr)))?
            .recv()
            .await
    }
}

/// Handle to one TCP connection. Clones share the connection.
///
/// Reads and writes lock separate halves of the stream, so one task can keep
/// sending while another waits for replies. A caller that expects replies in
/// order holds [`PeerConnection::reader`] for as long as it has requests
/// outstanding, so no other task takes its replies.
///
/// A read or write that fails or is cancelled part-way leaves the stream out of
/// step with the peer. The connection is then marked broken, and every later
/// read or write on it fails at once instead of misreading the stream.
#[derive(Debug, Clone)]
pub struct PeerConnection {
    pub addr: SocketAddr,
    reader: Arc<Mutex<OwnedReadHalf>>,
    writer: Arc<Mutex<OwnedWriteHalf>>,
    /// Content hashes of blobs the peer is believed to hold for this connection
    /// (see `TcpPacket::WORKREF`). Shared by every handle to the connection and
    /// reset when it is replaced.
    known_blobs: Arc<std::sync::Mutex<HashSet<u64>>>,
    broken: Arc<AtomicBool>,
}

/// Exclusive use of a connection's read half; see [`PeerConnection::reader`].
pub struct PeerReader<'a> {
    conn: &'a PeerConnection,
    half: MutexGuard<'a, OwnedReadHalf>,
}

impl PeerReader<'_> {
    /// Receives the next length-prefixed packet.
    pub async fn recv(&mut self) -> Result<TcpPacket, StruumError> {
        let addr = self.conn.addr;
        let half = &mut *self.half;
        self.conn.guarded(read_packet(half, addr)).await
    }
}

/// Marks the connection broken when dropped before [`OpGuard::done`]: the
/// operation failed or was cancelled mid-frame.
struct OpGuard<'a>(&'a AtomicBool, bool);

impl OpGuard<'_> {
    fn done(mut self) {
        self.1 = true;
    }
}

impl Drop for OpGuard<'_> {
    fn drop(&mut self) {
        if !self.1 {
            self.0.store(true, Ordering::SeqCst);
        }
    }
}

impl PeerConnection {
    /// Locks the set of blob hashes the peer is believed to hold.
    pub fn known_blobs(&self) -> std::sync::MutexGuard<'_, HashSet<u64>> {
        self.known_blobs.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether an earlier read or write failed or was cancelled part-way.
    pub fn is_broken(&self) -> bool {
        self.broken.load(Ordering::SeqCst)
    }

    async fn guarded<T>(
        &self,
        op: impl std::future::Future<Output = Result<T, StruumError>>,
    ) -> Result<T, StruumError> {
        if self.is_broken() {
            return Err(NetworkConnectionError(format!(
                "connection to {} is broken by an earlier failed read or write",
                self.addr
            )));
        }
        let guard = OpGuard(&self.broken, false);
        let out = op.await;
        if out.is_ok() {
            guard.done();
        }
        out
    }

    /// Waits for exclusive use of the read half. Replies arrive in the order
    /// requests were sent, so hold it from the first request to the last reply.
    pub async fn reader(&self) -> PeerReader<'_> {
        PeerReader { conn: self, half: self.reader.lock().await }
    }

    /// Sends one length-prefixed packet.
    pub async fn send(&self, packet: &TcpPacket) -> Result<(), StruumError> {
        let mut writer = self.writer.lock().await;
        self.guarded(write_packet(&mut *writer, self.addr, packet)).await
    }

    /// Sends a packet whose last field is a byte vector (e.g. `WORK`): `header`
    /// is the packet with that field left empty, and `payload` is written
    /// straight after it, so large payloads are never copied into a frame. The
    /// bytes on the wire are identical to sending the full packet.
    pub async fn send_with_payload(&self, header: &TcpPacket, payload: &[u8]) -> Result<(), StruumError> {
        let frame = Self::frame_with_payload(header, payload.len())?;
        let mut writer = self.writer.lock().await;
        let addr = self.addr;
        self.guarded(async {
            let w = &mut *writer;
            w.write_all(&frame).await.map_err(|e| NetworkConnectionError(e.to_string()))?;
            w.write_all(payload).await.map_err(|e| NetworkConnectionError(e.to_string()))?;
            w.flush().await.map_err(|e| NetworkConnectionError(e.to_string()))?;
            log::debug!("Sent TCP packet ({} bytes) to {}", frame.len() - 4 + payload.len(), addr);
            Ok(())
        })
        .await
    }

    /// Receives one length-prefixed packet.
    pub async fn recv(&self) -> Result<TcpPacket, StruumError> {
        self.reader().await.recv().await
    }

    /// Sends `packet` and waits for its reply, holding the read half throughout.
    pub async fn request(&self, packet: &TcpPacket) -> Result<TcpPacket, StruumError> {
        let mut reader = self.reader().await;
        self.send(packet).await?;
        reader.recv().await
    }

    /// [`PeerConnection::send_with_payload`] followed by waiting for the reply.
    pub async fn request_with_payload(
        &self,
        header: &TcpPacket,
        payload: &[u8],
    ) -> Result<TcpPacket, StruumError> {
        let mut reader = self.reader().await;
        self.send_with_payload(header, payload).await?;
        reader.recv().await
    }

    /// Length prefix + `header`, with the trailing empty byte field's length set to `payload_len`.
    pub fn frame_with_payload(header: &TcpPacket, payload_len: usize) -> Result<Vec<u8>, StruumError> {
        let mut frame = vec![0u8; 4];
        bincode::serialize_into(&mut frame, header).map_err(|e| NetworkConnectionError(e.to_string()))?;
        // bincode ends the packet with the empty field's u64 length (0): patch it.
        let len_at = frame.len() - 8;
        if frame[len_at..] != [0u8; 8] {
            return Err(NetworkConnectionError(
                "header packet must end with an empty byte field".to_string(),
            ));
        }
        frame[len_at..].copy_from_slice(&(payload_len as u64).to_le_bytes());
        let total = u32::try_from(frame.len() - 4 + payload_len)
            .map_err(|_| NetworkConnectionError(format!("packet too large: {payload_len} bytes")))?;
        frame[..4].copy_from_slice(&total.to_be_bytes());
        Ok(frame)
    }
}

async fn write_packet(
    stream: &mut (impl AsyncWrite + Unpin),
    addr: SocketAddr,
    packet: &TcpPacket,
) -> Result<(), StruumError> {
    let payload_len = bincode::serialized_size(packet)
        .map_err(|e| NetworkConnectionError(e.to_string()))? as usize;
    // Header and payload in one buffer => one write, one TCP segment burst.
    let mut frame = Vec::with_capacity(4 + payload_len);
    frame.extend_from_slice(&(payload_len as u32).to_be_bytes());
    bincode::serialize_into(&mut frame, packet).map_err(|e| NetworkConnectionError(e.to_string()))?;
    stream
        .write_all(&frame)
        .await
        .map_err(|e| NetworkConnectionError(e.to_string()))?;
    stream
        .flush()
        .await
        .map_err(|e| NetworkConnectionError(e.to_string()))?;
    log::debug!("Sent TCP packet ({} bytes) to {}", payload_len, addr);
    Ok(())
}

async fn read_packet(stream: &mut (impl AsyncRead + Unpin), addr: SocketAddr) -> Result<TcpPacket, StruumError> {
    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| NetworkConnectionError(format!("Failed to read TCP length header from {}: {}", addr, e)))?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len > MAX_FRAME_BYTES {
        return Err(NetworkConnectionError(format!(
            "{addr} sent a {len}-byte packet (limit {MAX_FRAME_BYTES})"
        )));
    }

    let mut payload = vec![0u8; len];
    stream
        .read_exact(&mut payload)
        .await
        .map_err(|e| NetworkConnectionError(format!("Failed to read TCP payload from {}: {}", addr, e)))?;

    let packet: TcpPacket = bincode::deserialize(&payload)
        .map_err(|e| StruumError::SerializationError(format!("Failed to deserialize TcpPacket from {}: {}", addr, e)))?;
    log::debug!("Received TCP packet ({} bytes) from {}", len, addr);
    Ok(packet)
}
