use std::net::UdpSocket;
use struum_types::StruumError;
use struum_types::network::{HelloPacket, UID};

pub struct Worker {
    pub id: UID,
}

const PORT:u16 = 34254;

#[allow(dead_code)]
impl Worker {
    pub fn notify_network(&mut self) -> Result<(), StruumError> {
        let socket = UdpSocket::bind("0.0.0.0:0");
        if let Ok(socket) = socket {
            let data = HelloPacket{};
            let mut buf = bincode::serialize(&data).map_err(|_| StruumError::NetworkConnectionError("Failed to serialize packet!"))?;
            socket
                .set_broadcast(true)
                .map_err(|_| StruumError::NetworkConnectionError("Couldn't enable broadcast!".to_string()))?;

            socket
                .send_to(&mut buf, format!("255.255.255.255:{}",PORT))
                .map_err(|_| {
                    StruumError::NetworkConnectionError("Couldn't receive a broadcast connection!".to_string())
                })?;

            Ok(())
        } else {
            Err(StruumError::NetworkConnectionError("Failed to broadcast!".to_string()))
        }
    }
    pub fn listen(&self) -> Result<(), StruumError> {
        let socket = UdpSocket::bind(format!("0.0.0.0:{}",PORT));
        if let Ok(socket) = socket {
            let mut buf = [0; 10];

            let (amt,src) = socket
                .recv_from(&mut buf)
                .map_err(|_| {
                    StruumError::NetworkConnectionError("Couldn't receive a broadcast connection!".to_string())
                })?;
            let message = &buf[..amt];
            println!("Received {:?} from {:?}", message, src);
            Ok(())
        } else {
            Err(StruumError::NetworkConnectionError("Failed to broadcast!".to_string()))
        }
    }
}
