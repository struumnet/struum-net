use serde::Deserialize;
use std::collections::HashMap;
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
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::Mutex;

/// Network layer is an Abstract layers that allows for any higher-level objects
/// like `Node`, `Coordinator` to consist of a Network Layer for itself.
#[derive(Debug)]
pub struct NetworkLayer<const BUF_SIZE: usize> {
    /// Public(Local) IPv4 address of itself, fetched during an outbound UDP connection.
    pub ip: IpAddr,
    /// A pool of connections of SocketAddresses with live connection made with other `NetworkLayer`s.
    connections: HashMap<SocketAddr, Arc<Mutex<TcpStream>>>,
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
        // Request/response traffic: don't let Nagle + delayed ACK stall small packets.
        let _ = stream.set_nodelay(true);

        self.connections
            .insert(target_addr, Arc::new(Mutex::new(stream)));

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
        self.connections.get(addr).map(|stream| PeerConnection {
            addr: *addr,
            stream: stream.clone(),
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
        let _ = stream.set_nodelay(true);
        log::info!("Accepted TCP connection from {}", peer_addr);
        self.connections
            .insert(peer_addr, Arc::new(Mutex::new(stream)));
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

/// Handle to one TCP connection. Cloning it shares the same stream, so a
/// request/response exchange should hold it for both halves (see [`PeerConnection::request`]).
#[derive(Debug, Clone)]
pub struct PeerConnection {
    pub addr: SocketAddr,
    stream: Arc<Mutex<TcpStream>>,
}

impl PeerConnection {
    /// Sends one length-prefixed packet.
    pub async fn send(&self, packet: &TcpPacket) -> Result<(), StruumError> {
        let mut stream = self.stream.lock().await;
        Self::write_packet(&mut stream, self.addr, packet).await
    }

    /// Receives one length-prefixed packet.
    pub async fn recv(&self) -> Result<TcpPacket, StruumError> {
        let mut stream = self.stream.lock().await;
        Self::read_packet(&mut stream, self.addr).await
    }

    /// Sends `packet` and waits for the reply while holding the connection.
    pub async fn request(&self, packet: &TcpPacket) -> Result<TcpPacket, StruumError> {
        let mut stream = self.stream.lock().await;
        Self::write_packet(&mut stream, self.addr, packet).await?;
        Self::read_packet(&mut stream, self.addr).await
    }

    async fn write_packet(
        stream: &mut TcpStream,
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

    async fn read_packet(stream: &mut TcpStream, addr: SocketAddr) -> Result<TcpPacket, StruumError> {
        let mut len_buf = [0u8; 4];
        stream
            .read_exact(&mut len_buf)
            .await
            .map_err(|e| NetworkConnectionError(format!("Failed to read TCP length header from {}: {}", addr, e)))?;
        let len = u32::from_be_bytes(len_buf) as usize;

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
}
