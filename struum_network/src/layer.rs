use serde::Deserialize;
use std::collections::HashMap;
use std::fmt::Debug;
use std::net::{IpAddr, SocketAddr};
use std::net::AddrParseError;
use std::sync::Arc;
use struum_types::StruumError::{NetworkConnectionError,ParserError};
use struum_types::{
    StruumError,
    network::{TcpPacket, UdpPacket},
};
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpSocket, TcpStream, UdpSocket};
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
}

impl<const BUF_SIZE: usize> NetworkLayer<BUF_SIZE> {
    pub async fn new(udp_port: u16, tcp_port: u16) -> Result<Self, StruumError> {
            Ok(Self {
                ip: get_local_ip().await?,
                connections: HashMap::default(),
                udp_port,
                tcp_port,
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
        .map_err(|e|{StruumError::NetworkConnectionError(e.to_string())})?;

    Ok(socket
        .local_addr()
        .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?
        .ip())
}

impl<const BUF_SIZE: usize> NetworkLayer<BUF_SIZE> {
    /// This methods allows for the nodes to initialize a targeted tcp connection within their own
    /// `NetworkLayer<T>`. To communicate directly with the node.
    pub async fn initialize_tcp_connection(
        &mut self,
        target_ip: &str,
        target_port: u16,
    ) -> Result<(), StruumError> {
        let self_addr = format!("0.0.0.0:{}", self.tcp_port)
            .parse()
            .map_err(|e: AddrParseError| StruumError::NetworkConnectionError(e.to_string()))?;
        let target_addr = format!("{}:{}", target_ip, target_port)
            .parse()
            .map_err(|e: AddrParseError| StruumError::NetworkConnectionError(e.to_string()))?;
        let socket =
            TcpSocket::new_v4().map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
        socket
            .bind(self_addr)
            .map_err(|e| NetworkConnectionError(e.to_string()))?;

        let conn = socket
            .connect(target_addr)
            .await
            .map_err(|e| NetworkConnectionError(e.to_string()))?;
        self.connections
            .insert(target_addr, Arc::new(Mutex::new(conn)));

        Ok(())
    }

    /// Broadcasts a UDP packet to the local devices
    pub async fn broadcast(&mut self, packet: UdpPacket) -> Result<(), StruumError> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
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
        socket.set_broadcast(true).map_err(|_| {
            StruumError::NetworkConnectionError("Couldn't enable broadcast!".to_string())
        })?;

        socket
            .send_to(&mut buf, format!("255.255.255.255:{}", self.udp_port))
            .await
            .map_err(|_| {
                StruumError::NetworkConnectionError(
                    "Couldn't receive a broadcast connection!".to_string(),
                )
            })?;

        Ok(())
    }

    pub async fn send_udp_data(
        &mut self,
        packet: UdpPacket,
        addr: SocketAddr,
    ) -> Result<(), StruumError> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
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

        socket.send_to(&mut buf, addr).await.map_err(|_| {
            StruumError::NetworkConnectionError(
                "Couldn't receive a broadcast connection!".to_string(),
            )
        })?;

        Ok(())
    }

    /// This method is used for node to listen for any UDP Packets
    pub async fn listen_udp<T: Debug + for<'a> Deserialize<'a>>(
        &mut self,
    ) -> Result<(T, SocketAddr), StruumError> {
        let socket = UdpSocket::bind(format!("{}:{}",self.ip, self.udp_port))
            .await
            .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
        let mut buf = [0; BUF_SIZE];

        let (amt, src) = socket.recv_from(&mut buf).await.map_err(|_| {
            StruumError::NetworkConnectionError(
                "Couldn't receive a broadcast connection!".to_string(),
            )
        })?;
        let message = bincode::deserialize::<T>(&buf[..amt])
            .map_err(|e| StruumError::SerializationError(e.to_string()))?;
        println!("Received {:?} from {:?}", message, src);
        Ok((message, src))
    }

    /// This methods allows for the nodes to entirely all the data, to the node tcp connection
    /// was initialized using `initialize_connectionsnection()`.
    pub async fn send_tcp_data(
        &mut self,
        addr: SocketAddr,
        data: TcpPacket,
    ) -> Result<(), StruumError> {
        let data_serialized =
            bincode::serialize(&data).map_err(|e| NetworkConnectionError(e.to_string()))?;
        let connection = self
            .connections
            .get(&addr)
            .ok_or(NetworkConnectionError(
                "Failed to find existing connection between nodes!".to_string(),
            ))?
            .clone();
        let mut stream = connection.lock().await;
        stream
            .write_all(&data_serialized)
            .await
            .map_err(|e| NetworkConnectionError(e.to_string()))?;
        Ok(())
    }
}
