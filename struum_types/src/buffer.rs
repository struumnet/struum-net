#[derive(Debug)]
pub enum BufferRole {
    Input,
    Output,
}

#[derive(Debug)]
pub struct Buffer {
    pub role: BufferRole,
    pub data: Vec<u8>,
}
