#[derive(Debug, Clone, Copy)]
pub enum BufferRole {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub role: BufferRole,
    pub data: Vec<u8>,
    pub stride: usize,
}

impl Buffer {
    pub fn new<T: bytemuck::Pod>(role: BufferRole, data: &[T]) -> Self {
        Self {
            role: role,
            data: bytemuck::cast_slice(data).to_vec(),
            stride: std::mem::size_of::<T>(),
        }
    }

    pub fn empty<T: bytemuck::Pod>(role: BufferRole, count: usize) -> Self {
        Self {
            role: role,
            data: vec![0u8; count * std::mem::size_of::<T>()],
            stride: std::mem::size_of::<T>(),
        }
    }

    pub fn as_slice<T: bytemuck::Pod>(&self) -> &[T] {
        bytemuck::cast_slice(&self.data)
    }

    /// Returns the length of data / stride
    /// If buffer is of f32 then it returns len(data) / sizeof(f32)
    pub fn actual_len(&self) -> usize {
        self.data.len() / self.stride
    }
}
