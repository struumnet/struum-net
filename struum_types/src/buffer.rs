#[derive(Debug, Clone)]
pub enum BufferRole {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub role: BufferRole,
    pub data: Vec<u8>,
}

impl Buffer {
    pub fn new(role: BufferRole, data: Vec<u8>) -> Self {
        Self {
            role,
            data,
        }
    }
}
