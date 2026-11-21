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
    TASKDATA(TaskDataPacket),
    RESULT(ResultPacket),
    // New variants are appended so existing bincode tags stay stable.
    WORK(WorkPacket),
    WORKERROR(WorkErrorPacket),
}

/// Kind of work carried by a [`WorkPacket`].
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    /// Serialized GLSL `Kernel`.
    Gpu,
    /// Serialized WebAssembly `CpuTask`.
    Cpu,
}

/// A unit of work (possibly one shard of a larger job). Answered with
/// `RESULT` (same serialized type) or `WORKERROR`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WorkPacket {
    pub task_id: String,
    pub sender_id: UID,
    pub kind: WorkKind,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

/// Reports that a worker failed to execute a [`WorkPacket`] (bad shader, wasm trap, ...).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct WorkErrorPacket {
    pub task_id: String,
    pub sender_id: UID,
    pub message: String,
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
pub struct TaskDataPacket {
    /// ID of the task/job.
    pub task_id: String,
    /// ID of the node that sent this task.
    pub sender_id: UID,
    /// Serialized kernel data (bincode-encoded Kernel struct).
    #[serde(with = "serde_bytes")]
    pub kernel_data: Vec<u8>,
}

impl TaskDataPacket {
    pub fn new(task_id: String, sender_id: UID, kernel_data: Vec<u8>) -> Self {
        Self { task_id, sender_id, kernel_data }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ResultPacket {
    /// ID of the job that produced this result.
    pub job_id: String,
    /// ID of the node that computed the result.
    pub sender_id: UID,
    /// Serialized result data.
    #[serde(with = "serde_bytes")]
    pub result_data: Vec<u8>,
}

impl ResultPacket {
    pub fn new(job_id: String, sender_id: UID, result_data: Vec<u8>) -> Self {
        Self { job_id, sender_id, result_data }
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
    #[serde(default)]
    pub tcp_port: Option<u16>,
}

impl From<IntroductionPacket> for NodeDetails {
    fn from(packet: IntroductionPacket) -> Self {
        NodeDetails {
            id: packet.id,
            ip: packet.ip,
            role: packet.role,
            backend: packet.backend,
            tcp_port: packet.tcp_port,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct NodeDetails {
    pub id: UID,
    pub ip: SocketAddr,
    pub role: NodeRole,
    pub backend: Option<NodeBackend>,
    #[serde(default)]
    pub tcp_port: Option<u16>,
}

impl NodeDetails {
    /// Returns the SocketAddr for reaching this node over TCP, if tcp_port is known.
    pub fn tcp_addr(&self) -> Option<SocketAddr> {
        self.tcp_port.map(|p| SocketAddr::new(self.ip.ip(), p))
    }
}

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_UID: AtomicU64 = AtomicU64::new(1);

/// Node identifier. 64-bit and randomly generated, so independently started
/// nodes practically never collide.
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
    pub inner: u64,
}

impl UID {
    pub fn new(inner: u64) -> UID {
        UID { inner }
    }

    /// Generates a random, non-zero UID. Uses the process-random `RandomState`
    /// keys mixed with time, pid and a counter, so no extra RNG dependency is needed.
    pub fn generate() -> UID {
        use std::hash::{BuildHasher, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        h.write_u32(std::process::id());
        h.write_u64(NEXT_UID.fetch_add(1, Ordering::Relaxed));
        let val = h.finish();
        UID { inner: if val == 0 { 1 } else { val } }
    }

    pub fn id(&self) -> u64 {
        self.inner
    }
}

impl From<u64> for UID {
    fn from(inner: u64) -> Self {
        UID { inner }
    }
}

impl From<UID> for u64 {
    fn from(uid: UID) -> Self {
        uid.inner
    }
}

impl std::ops::Deref for UID {
    type Target = u64;

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

    /// Initiates information exchange between registered nodes.
    async fn establish_information_exchange(&mut self) -> Result<(), StruumError>;
}

#[cfg(test)]
mod tests {
    use super::UID;
    use std::collections::HashSet;

    #[test]
    fn generated_uids_are_unique_and_non_zero() {
        let ids: HashSet<UID> = (0..10_000).map(|_| UID::generate()).collect();
        assert_eq!(ids.len(), 10_000);
        assert!(!ids.contains(&UID::new(0)));
    }
}
