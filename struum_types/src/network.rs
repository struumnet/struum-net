use serde_derive::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr};
/// This enum describes the state of the connection that the worker is in,
///
/// CURRENTLY IN THE MVP
/// By Default, a worker is in a polling state to the router
/// unless it receives a
#[derive(Debug)]
pub enum NodeConnectionState {
    POLLING,
    FOUND,
    CONNECTING,
    CONNECTED,
}

/// This enum is used to describe the state of the worker in a network
#[derive(Debug)]
pub enum NodeState {
    IDLE,
    WORKING,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum UdpPacket {
    HELLO(HelloPacket),
    INTRODUCTION(IntroductionPacket),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum TcpPacket {
    TASK(TaskPacket),
    REGISTERTASK(RegisterTaskPacket),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterTaskPacket {
    input: u8, // PLACEHOLDER
    task_inner: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TaskPacket {
    id: UID, // PLACEHOLDER
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HelloPacket {}

#[derive(Serialize, Deserialize, Debug)]
pub struct IntroductionPacket {
    pub ip: SocketAddr,
    pub role: NodeRole,
    pub backend: NodeBackend,
}

#[derive(Default, Deserialize, Serialize, Debug)]
pub struct UID {
    inner: u8,
}

impl UID {
    pub fn new(inner: u8) -> UID {
        return UID { inner };
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub enum NodeBackend {
    OpenGL,
    Vulkan,
    Cpu,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub enum NodeRole {
    COORDINATOR,
    NODE,
}
