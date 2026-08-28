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

    pub async fn introduce(&mut self) -> Result<(), StruumError> {
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    ip: SocketAddr::new(self.net.ip, self.net.udp_port),
                    role: NodeRole::NODE,
                    backend: Some(self.backend),
                }),
                SocketAddr::new(self.net.ip, self.net.udp_port),
            )
            .await?;
        Ok(())
    }

    pub async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError> {
        let (resp, src) = self.net.listen_udp::<HelloPacket>().await?;
        println!("Hello Received!");
        return Ok(src);
    }
    pub async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError> {
        let (resp, _) = self.net.listen_udp::<IntroductionPacket>().await?;
        println!("Introduction Received!");
        return Ok(resp);
    }
}
