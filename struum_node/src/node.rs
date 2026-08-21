use struum_network::layer::NetworkLayer;
use struum_types::StruumError;
use struum_types::network::{HelloPacket, UID,Packet};

pub struct Node<const BUF_SIZE: usize> {
    pub id: UID,
    pub net: NetworkLayer<BUF_SIZE>,
}

impl<const BUF_SIZE: usize> Node<BUF_SIZE> {
    pub async fn notify_network(&mut self) -> Result<(), StruumError> {
        return self.net.broadcast(Packet::HELLO(HelloPacket{})).await;
    }
    pub async fn listen_hello(&mut self) -> Result<(), StruumError> {
        return self.net.listen::<HelloPacket>().await;
    }
}
