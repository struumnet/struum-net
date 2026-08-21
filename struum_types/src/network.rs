use serde_derive::{Deserialize, Serialize};
/// This enum describes the state of the connection that the worker is in,
///
/// CURRENTLY IN THE MVP
/// By Default, a worker is in a polling state to the router
/// unless it receives a
#[derive(Debug)]
pub enum WorkerConnectionState {
    POLLING,
    FOUND,
    CONNECTING,
    CONNECTED,
}
/// This enum is used to describe the state of the worker in a network
#[derive(Debug)]
pub enum WorkerState {
    IDLE,
    WORKING,
}

pub enum Packet {
    TASK_REGISTER(RegisterTaskPacket),
    TASK(TaskPacket),
    HELLO(HelloPacket),
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

#[derive(Default, Deserialize, Serialize, Debug)]
pub struct UID {
    inner: u8,
}

impl UID {
    pub fn new(inner: u8) -> UID {
        return UID { inner };
    }
}
#[derive(Debug)]
pub enum WorkerBackend {
    OpenGL,
    Vulkan,
    Cpu,
}
