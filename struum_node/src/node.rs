use std::net::SocketAddr;
use struum_network::layer::NetworkLayer;
use struum_types::StruumError::{self};
use struum_types::network::{
    HelloPacket, IntroductionPacket, NodeBackend, NodeRole, UID, UdpPacket, NetworkCommunicator
};
use async_trait::async_trait;


#[derive(Debug)]
pub struct Node<const BUF_SIZE: usize> {
    /// Id is immuntable and its type shouldn't be assumed.
    pub id: UID,
    /// net uses defines a NetworkLayer with a BUF_SIZE for the TCP packet.
    pub net: NetworkLayer<BUF_SIZE>,
    /// backend defines the computation backend.
    pub backend: NodeBackend,
}

#[async_trait]
impl<const BUF_SIZE: usize> NetworkCommunicator for Node<BUF_SIZE> {
    /// LAN Broadcast for testing purposes, it broadcasts the HelloPacket introducing itself with
    /// its IP.
    async fn notify_network(&mut self) -> Result<(), StruumError> {
        return self.net.broadcast(UdpPacket::HELLO(HelloPacket {})).await;
    }
    /// This sends a unicast UDP packet to the receiving IP from the broadcast IP.
    async fn introduce(&mut self,ip: SocketAddr) -> Result<(), StruumError> {
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    ip: SocketAddr::new(self.net.ip, self.net.udp_port),
                    role: NodeRole::NODE,
                    backend: Some(self.backend),
                }),
                ip
            )
            .await?;
        Ok(())
    }

    /// Listens for hello packet for testing
    async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError> {
        let (_, src) = self.net.listen_udp::<HelloPacket>().await?;
        println!("Hello Received!");
        return Ok(src);
    }

    /// Listens for introduction packet from UDP unicast packet.
    async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError> {
        let (resp, _) = self.net.listen_udp::<IntroductionPacket>().await?;
        println!("Introduction Received!");
        return Ok(resp);
    }
}
