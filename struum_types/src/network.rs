use serde_derive::{Serialize,Deserialize};

/// This enum describes the state of the connection that the worker is in,
///
/// CURRENTLY IN THE MVP
/// By Default, a worker is in a polling state to the router
/// unless it receives a
pub enum WorkerConnectionState {
    POLLING,
    FOUND,
    CONNECTING,
    CONNECTED,
}
/// This enum is used to describe the state of the worker in a network
pub enum WorkerState {
    IDLE,
    WORKING,
}


#[derive(Serialize,Deserialize)]
pub struct RegisterTaskPacket {
    input: u8, // PLACEHOLDER
    task_inner: String,
}

#[derive(Serialize,Deserialize)]
pub struct TaskPacket {
    id: UID, // PLACEHOLDER
}

#[derive(Serialize,Deserialize)]
pub struct HelloPacket {}

#[derive(Default,Deserialize,Serialize)]
pub struct UID{
    inner:u8
}

impl UID {
    pub fn new(inner:u8) -> UID {
        return UID{
            inner
        };
    }
}

pub enum WorkerBackend {
    OpenGL,
    Vulkan,
    Cpu,
}
