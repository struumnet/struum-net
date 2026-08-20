#[derive(Debug, Clone)]
pub struct Buffer {
    pub name: String,
    pub data: Vec<u8>,
    pub stride: usize,
}

impl Buffer {
    pub fn new<T: bytemuck::Pod>(name: &str, data: &[T]) -> Self {
        Self {
            name: name.to_string(),
            data: bytemuck::cast_slice(data).to_vec(),
            stride: std::mem::size_of::<T>(),
        }
    }

    pub fn empty<T: bytemuck::Pod>(name: &str, count: usize) -> Self {
        Self {
            name: name.to_string(),
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
