use std::net::SocketAddr;

use struum_network::layer::NetworkLayer;
use struum_types::StruumError::{self};
use struum_types::network::{
    HelloPacket, IntroductionPacket, NodeBackend, NodeRole, UID, UdpPacket,
};

#[derive(Debug)]
pub struct Node<const BUF_SIZE: usize> {
    pub id: UID,
    pub net: NetworkLayer<BUF_SIZE>,
    pub backend: NodeBackend,
}

impl<const BUF_SIZE: usize> Node<BUF_SIZE> {
    pub async fn notify_network(&mut self) -> Result<(), StruumError> {
        return self.net.broadcast(UdpPacket::HELLO(HelloPacket {})).await;
    }

    pub async fn introduce_node(&mut self, addr: SocketAddr) -> Result<(), StruumError> {
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    ip: SocketAddr::new(self.net.ip, self.net.udp_port),
                    role: NodeRole::NODE,
                    backend: self.backend,
                }),
                addr,
            )
            .await?;
        Ok(())
    }

    pub async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError> {
        let (resp, src) = self.net.listen_udp::<UdpPacket>().await?;
        if let UdpPacket::HELLO(p) = resp {
            println!("Hello Received!");
            return Ok(src);
        }
        return Err(StruumError::NetworkConnectionError(
            "Failed to received HELLO".to_string(),
        ));
    }
    pub async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError> {
        let (resp, _) = self.net.listen_udp::<UdpPacket>().await?;
        if let UdpPacket::INTRODUCTION(intro) = resp {
            println!("Introduction Received!");
            return Ok(intro);
        }
        return Err(StruumError::NetworkConnectionError(
            "Failed to received HELLO".to_string(),
        ));
    }
}
