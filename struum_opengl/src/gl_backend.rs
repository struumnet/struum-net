use crate::gl_context::GlContext;
use crate::shader::Shader;
use crate::gl_buffer::GlBuffer;

use std::collections::HashMap;

use struum_types::{
    StruumError,
    Buffer,
};

pub struct OpenglBackend {
    shader: Shader,
    gl_buffers: HashMap<String, GlBuffer>,
    _context: GlContext,
}

impl OpenglBackend {
    pub fn new(shader: &str, buffers: &[Buffer]) -> Result<OpenglBackend, StruumError> {
        let context = GlContext::new()?;

        // Compile Shader
        let shader = Shader::new(shader)?;

        // Construct opengl buffers
        let mut gl_buffers = HashMap::new();

        for (binding, buffer) in buffers.iter().enumerate() {
            let gl_buffer = GlBuffer::new(buffer);
            gl_buffer.bind(binding as u32);
            gl_buffers.insert(gl_buffer.name.clone(), gl_buffer);
        }

        Ok(Self {
            shader: shader,
            gl_buffers: gl_buffers,
            _context: context,
        })
    }

    pub fn execute(&self) {

        // TODO(slok): Determine dispatch size.
        // (input_count + invocation_count - 1) / invocation_count
        self.shader.dispatch_and_wait(1, 1, 1);
    }

    pub fn update(&mut self, buffer_name: &str, buffer: &Buffer) -> Result<(), StruumError> {
        let gl_buffer = self
            .gl_buffers
            .get_mut(buffer_name)
            .ok_or_else(|| {
                StruumError::NotFound(buffer_name.to_string())
            })?;

        gl_buffer.write(&buffer.data)?;

        Ok(())
    }

    pub fn read_buffer<T: bytemuck::Pod>(&mut self, buffer_name: &str) -> Result<Vec<T>, StruumError> {
        let gl_buffer = self
            .gl_buffers
            .get_mut(buffer_name)
            .ok_or_else(|| {
                StruumError::NotFound(buffer_name.to_string())
            })?;

            let data = gl_buffer.read();

        Ok(bytemuck::cast_slice(&data).to_vec())
    }
}
