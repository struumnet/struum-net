use struum_types::{StruumError, network::{HelloPacket, IntroductionPacket, NodeRole, UID}};
use struum_network::layer::NetworkLayer;

/// Represents the Coordinator for connections
pub struct Coordinator<T> {
    pub _id: UID,
    net: NetworkLayer::<T>,
}

impl Coordinator {
    pub fn new(udp_port: u16, tcp_port: u16)->Result<Self,StruumError>{
        Ok(Self{
            _id: UID::new(0),
            net: NetworkLayer::new(udp_port,tcp_port)?,
        })
    }
}

impl Coordinator {
    pub async fn notify_network(&mut self) -> Result<(), StruumError> {
        return self.net.broadcast(UdpPacket::HELLO(HelloPacket {})).await;
    }

    pub async fn introduce(&mut self, addr: SocketAddr) -> Result<(), StruumError> {
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    ip: SocketAddr::new(self.net.ip, self.net.udp_port),
                    role: NodeRole::COORDINATOR,
                    backend: None,
                }),
                addr,
            )
            .await?;
        Ok(())
    }

    pub async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError> {
        let (resp, src) = self.net.listen_udp::<HelloPacket>().await?;
        if let UdpPacket::HELLO(p) = resp {
            println!("Hello Received!");
            return Ok(src);
        }
        return Err(StruumError::NetworkConnectionError(
            "Failed to received HELLO".to_string(),
        ));
    }
    pub async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError> {
        let (resp, _) = self.net.listen_udp::<IntroductionPacket>().await?;
        if let UdpPacket::INTRODUCTION(intro) = resp {
            println!("Introduction Received!");
            return Ok(intro);
        }
        return Err(StruumError::NetworkConnectionError(
            "Failed to received HELLO".to_string(),
        ));
    }
}
