use serde_derive::{Deserialize, Serialize};
use std::net::{SocketAddr};
use async_trait::async_trait;

use crate::StruumError;
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
    pub backend: Option<NodeBackend>,
}

impl From<IntroductionPacket> for NodeDetails {
    fn from(packet: IntroductionPacket) -> Self {
        NodeDetails { ip: packet.ip, role: packet.role, backend: packet.backend }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NodeDetails {
    pub ip: SocketAddr,
    pub role: NodeRole,
    pub backend: Option<NodeBackend>,
}

#[derive(Default, Deserialize, Serialize, Debug,PartialEq,Eq,Hash)]
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

#[async_trait]
pub trait NetworkCommunicator {
    /// This method is used to notify the network about its presence
    async fn notify_network(&mut self) -> Result<(), StruumError>;
    /// This method is used by a `Node` and sends `IntroductionPacket` to the `Coordinator`
    /// to register itself to the network.
    async fn introduce(&mut self,ip: SocketAddr) -> Result<(), StruumError>;


    // TODO: MIGHT HAVE TO IMPROVE ON THIS
    /// Hello request listener
    async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError>;
    /// Introduction listener
    async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError>;
}

#[async_trait]
pub trait NetworkCoordinator {
    /// This method is used if a node is missing the information of
    /// any sibling nodes, and the `Coordinator` introduces through
    /// information on the map.
    async fn introduce_sibling_node(&mut self,node_id: &UID) -> Result<&NodeDetails, StruumError>;

    /// This method is used to best described as a session initiation protocol
    /// E.g.
    ///
    /// PRE:
    ///       `Coordinator`
    ///      |           |
    /// `Node` A          `Node` B
    ///
    /// POST:
    /// `Node` A <------> `Node` B
    ///
    async fn establish_information_exchange(&mut self) -> Result<(), StruumError>;
}
