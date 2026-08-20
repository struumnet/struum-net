use std::ffi::c_void;
use struum_types::{StruumError, Buffer};

pub(crate) struct GlBuffer {
    pub name: String,
    pub id: u32,
    pub size: usize,
    pub stride: usize,
}

impl GlBuffer {
    pub(crate) fn new(buffer: &Buffer) -> Self {
        let mut id = 0;

        unsafe {
            gl::GenBuffers(1, &mut id);

            gl::BindBuffer(
                gl::SHADER_STORAGE_BUFFER,
                id,
            );

            gl::BufferData(
                gl::SHADER_STORAGE_BUFFER,
                buffer.data.len() as isize,
                buffer.data.as_ptr() as *const c_void,
                gl::DYNAMIC_COPY,
            );

            gl::BindBuffer(
                gl::SHADER_STORAGE_BUFFER,
                0,
            );
        }

        Self {
            name: buffer.name.clone(),
            id: id,
            size: buffer.data.len(),
            stride: buffer.stride,
        }
    }

    pub(crate) fn read(&self) -> Vec<u8> {
        let mut data = vec![0u8; self.size];

        unsafe {
            gl::BindBuffer(
                gl::SHADER_STORAGE_BUFFER,
                self.id,
            );

            gl::GetBufferSubData(
                gl::SHADER_STORAGE_BUFFER,
                0,
                self.size as isize,
                data.as_mut_ptr() as *mut c_void,
            );

            gl::BindBuffer(
                gl::SHADER_STORAGE_BUFFER,
                0,
            );
        }

        data
    }

    pub(crate) fn write(&mut self, data: &[u8]) -> Result<(), StruumError> {
        if data.len() > self.size {
            return Err(StruumError::BufferWriteError(
                "data is larger than the allocated GPU buffer".to_string()
            ));
        }

        unsafe {
            gl::BindBuffer(
                gl::SHADER_STORAGE_BUFFER,
                self.id,
            );

            gl::BufferSubData(
                gl::SHADER_STORAGE_BUFFER,
                0,
                data.len() as isize,
                data.as_ptr() as *const c_void,
            );

            gl::BindBuffer(
                gl::SHADER_STORAGE_BUFFER,
                0,
            );
        }

        Ok(())
    }

    pub(crate) fn bind(&self, binding: u32) {
        unsafe {
            gl::BindBufferBase(
                gl::SHADER_STORAGE_BUFFER,
                binding,
                self.id,
            );
        }
    }
}

impl Drop for GlBuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id);
        }
    }
}
