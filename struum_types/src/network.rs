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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum UdpPacket {
    HELLO(HelloPacket),
    INTRODUCTION(IntroductionPacket),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum TcpPacket {
    TASK(TaskPacket),
    REGISTERTASK(RegisterTaskPacket),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct RegisterTaskPacket {
    pub input: u8,
    pub task_inner: String,
}

impl RegisterTaskPacket {
    pub fn new(input: u8, task_inner: String) -> Self {
        Self { input, task_inner }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TaskPacket {
    pub id: UID,
}

impl TaskPacket {
    pub fn new(id: UID) -> Self {
        Self { id }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct HelloPacket {}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct IntroductionPacket {
    pub id: UID,
    pub ip: SocketAddr,
    pub role: NodeRole,
    pub backend: Option<NodeBackend>,
}

impl From<IntroductionPacket> for NodeDetails {
    fn from(packet: IntroductionPacket) -> Self {
        NodeDetails { id: packet.id, ip: packet.ip, role: packet.role, backend: packet.backend }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NodeDetails {
    pub id: UID,
    pub ip: SocketAddr,
    pub role: NodeRole,
    pub backend: Option<NodeBackend>,
}

use std::sync::atomic::{AtomicU8, Ordering};

static NEXT_UID: AtomicU8 = AtomicU8::new(1);

#[derive(
    Default,
    Deserialize,
    Serialize,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Clone,
    Copy,
)]
pub struct UID {
    pub inner: u8,
}

impl UID {
    pub fn new(inner: u8) -> UID {
        UID { inner }
    }

    /// Automatically generates a new unique UID.
    pub fn generate() -> UID {
        UID {
            inner: NEXT_UID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn id(&self) -> u8 {
        self.inner
    }
}

impl From<u8> for UID {
    fn from(inner: u8) -> Self {
        UID { inner }
    }
}

impl From<UID> for u8 {
    fn from(uid: UID) -> Self {
        uid.inner
    }
}

impl std::ops::Deref for UID {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::fmt::Display for UID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "UID({})", self.inner)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
pub enum NodeBackend {
    OpenGL,
    Vulkan,
    Cpu,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
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
    async fn introduce(&mut self, ip: SocketAddr) -> Result<(), StruumError>;

    /// Hello request listener
    async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError>;
    /// Introduction listener
    async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError>;
}

#[async_trait]
pub trait NetworkCoordinator {
    /// Looks up a registered node by ID in the coordinator.
    async fn introduce_sibling_node(&mut self, node_id: &UID) -> Result<&NodeDetails, StruumError>;

    /// Introduces two sibling nodes to each other by sending each node's introduction to the other.
    async fn introduce_nodes(&mut self, node_a_id: &UID, node_b_id: &UID) -> Result<(), StruumError>;

    /// Initiates information exchange between registered nodes (e.g. mesh introduction).
    async fn establish_information_exchange(&mut self) -> Result<(), StruumError>;
}
