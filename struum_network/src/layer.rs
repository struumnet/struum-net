use serde::Deserialize;
use struum_types::{StruumError, network::{Packet}};
use tokio::net::UdpSocket;
use std::fmt::Debug;

pub struct NetworkLayer<const BUF_SIZE: usize> {
    pub port: u16,
}

impl<const BUF_SIZE: usize> NetworkLayer<BUF_SIZE> {
    pub async fn broadcast(&mut self,packet: Packet) -> Result<(), StruumError> {
        let socket = UdpSocket::bind("0.0.0.0:0").await;
        if let Ok(socket) = socket {
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
                .send_to(&mut buf, format!("255.255.255.255:{}", self.port))
                .await
                .map_err(|_| {
                    StruumError::NetworkConnectionError(
                        "Couldn't receive a broadcast connection!".to_string(),
                    )
                })?;

            Ok(())
        } else {
            Err(StruumError::NetworkConnectionError(
                "Failed to broadcast!".to_string(),
            ))
        }
    }

    pub async fn listen<T: Debug + for<'a> Deserialize<'a>> (&mut self) -> Result<(), StruumError> {
        let socket = UdpSocket::bind(format!("0.0.0.0:{}", self.port)).await;
        if let Ok(socket) = socket {
            let mut buf = [0; BUF_SIZE];

            let (amt, src) = socket.recv_from(&mut buf).await.map_err(|_| {
                StruumError::NetworkConnectionError(
                    "Couldn't receive a broadcast connection!".to_string(),
                )
            })?;
            let message = bincode::deserialize::<T>(&buf[..amt]).map_err(|_| {
                StruumError::NetworkConnectionError("Failed to Serialize!".to_string())
            })?;
            println!("Received {:?} from {:?}", message, src);
            Ok(())
        } else {
            Err(StruumError::NetworkConnectionError(
                "Failed to broadcast!".to_string(),
            ))
        }
    }
}
