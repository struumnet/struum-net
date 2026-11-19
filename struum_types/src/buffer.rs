use serde_derive::{Deserialize, Serialize};
use std::ops::Range;

use crate::StruumError;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        if self.stride == 0 {
            return 0;
        }
        self.data.len() / self.stride
    }

    /// Returns a new buffer holding elements `range` (in units of `stride`).
    pub fn slice(&self, range: Range<usize>) -> Result<Buffer, StruumError> {
        if range.start > range.end || range.end > self.actual_len() {
            return Err(StruumError::ShardError(format!(
                "slice {:?} out of bounds for buffer '{}' with {} elements",
                range,
                self.name,
                self.actual_len()
            )));
        }
        Ok(Self {
            name: self.name.clone(),
            data: self.data[range.start * self.stride..range.end * self.stride].to_vec(),
            stride: self.stride,
        })
    }

    /// Concatenates buffers with the same name and stride, in order.
    pub fn concat(parts: &[Buffer]) -> Result<Buffer, StruumError> {
        let first = parts
            .first()
            .ok_or_else(|| StruumError::ShardError("cannot concat zero buffers".to_string()))?;
        if let Some(bad) = parts
            .iter()
            .find(|b| b.name != first.name || b.stride != first.stride)
        {
            return Err(StruumError::ShardError(format!(
                "cannot concat buffer '{}' (stride {}) with '{}' (stride {})",
                first.name, first.stride, bad.name, bad.stride
            )));
        }
        let mut data = Vec::with_capacity(parts.iter().map(|b| b.data.len()).sum());
        for part in parts {
            data.extend_from_slice(&part.data);
        }
        Ok(Self {
            name: first.name.clone(),
            data,
            stride: first.stride,
        })
    }
}
