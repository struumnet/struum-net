use serde::{Deserialize,Serialize};
use struum_types::StruumError::NetworkConnectionError;
use tokio::io::AsyncWriteExt;
use std::fmt::Debug;
use std::net::{AddrParseError};
use struum_types::{StruumError, network::Packet};
use tokio::net::{TcpSocket, TcpStream, UdpSocket};

pub struct NetworkLayer<const BUF_SIZE: usize> {
    udp_socket: UdpSocket,
    tcp_con: TcpStream,
    pub udp_port: u16,
}

impl<const BUF_SIZE: usize> NetworkLayer<BUF_SIZE> {

    /// This methods allows for the nodes to initialize a targeted tcp connection within their own
    /// `NetworkLayer<T>`. To communicate directly with the node.
    pub async fn initialize_tcp_connection(&mut self,target_ip:&str,target_port:u16) -> Result<(), StruumError> {
        let addr = format!("{}:{}",target_ip, target_port)
                    .parse()
                    .map_err(|e:AddrParseError| StruumError::NetworkConnectionError(e.to_string()))?;
        let socket =
            TcpSocket::new_v4().map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
        socket.bind(
            addr
        ).map_err(|e|{NetworkConnectionError(e.to_string())})?;

        self.tcp_con = socket.connect(addr).await.map_err(|e|{NetworkConnectionError(e.to_string())})?;
        Ok(())
    }
    pub async fn broadcast(&mut self, packet: Packet) -> Result<(), StruumError> {
        let socket = UdpSocket::bind("0.0.0.0:0")
            .await
            .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
        let mut buf: Vec<u8>;
        match packet {
            Packet::HELLO(data) => {
                buf = bincode::serialize(&data).map_err(|_| {
                    StruumError::NetworkConnectionError("Failed to serialize packet!".to_string())
                })?;
            }
            Packet::TASK(data) => {
                buf = bincode::serialize(&data).map_err(|_| {
                    StruumError::NetworkConnectionError("Failed to serialize packet!".to_string())
                })?;
            }
            Packet::TASK_REGISTER(data) => {
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
    /// This method is used for node to listen for any UDP Packets
    pub async fn listen<T: Debug + for<'a> Deserialize<'a>>(&mut self) -> Result<(), StruumError> {
        let socket = UdpSocket::bind(format!("0.0.0.0:{}", self.udp_port))
            .await
            .map_err(|e| StruumError::NetworkConnectionError(e.to_string()))?;
        let mut buf = [0; BUF_SIZE];

        let (amt, src) = socket.recv_from(&mut buf).await.map_err(|_| {
            StruumError::NetworkConnectionError(
                "Couldn't receive a broadcast connection!".to_string(),
            )
        })?;
        let message = bincode::deserialize::<T>(&buf[..amt])
            .map_err(|_| StruumError::NetworkConnectionError("Failed to Serialize!".to_string()))?;
        println!("Received {:?} from {:?}", message, src);
        Ok(())
    }

    /// This methods allows for the nodes to entirely all the data, to the node tcp connection
    /// was initialized using `initialize_tcp_connection()`.
    pub async fn send_tcp_data<T: Serialize + Debug>(&mut self,data:&T) -> Result<(), StruumError> {
        let data_serialized = bincode::serialize(data).map_err(|e|{NetworkConnectionError(e.to_string())})?;
        self.tcp_con
            .write_all(&data_serialized)
            .await
            .map_err(|e|NetworkConnectionError(e.to_string()))?;
        Ok(())
    }
}
