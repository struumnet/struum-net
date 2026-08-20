use crate::gl_context::GlContext;
use crate::shader::Shader;
use crate::gl_buffer::GlBuffer;

use struum_types::{
    StruumError,
    BufferRole,
    Buffer,
};

pub struct OpenglBackend {
  _context: GlContext,
}

impl OpenglBackend {
    pub fn new() -> Result<OpenglBackend, StruumError> {
        let context = GlContext::new()?;
        Ok(Self {
            _context: context,
        })
    }

    pub fn execute(&self, shader: &str, buffers: &[Buffer]) -> Result<Vec<Buffer>, StruumError> {
        let shader = Shader::new(shader)?;

        // Create opengl buffers
        let mut gl_buffers = Vec::with_capacity(buffers.len());

        for (binding, buffer) in buffers.iter().enumerate() {
            let gl_buffer = GlBuffer::new(buffer);
            gl_buffer.bind(binding as u32);
            gl_buffers.push(gl_buffer);
        }

        // TODO(slok): Determine dispatch size.
        // (input_count + invocation_count - 1) / invocation_count
        shader.dispatch_and_wait(1, 1, 1);

        // Read output buffers.
        let mut outputs = Vec::new();

        for (buffer, gl_buffer) in buffers.iter().zip(gl_buffers.iter()) {
            match buffer.role {
                BufferRole::Output => {
                    let data = gl_buffer.read();

                    outputs.push(Buffer {
                        role: BufferRole::Output,
                        data,
                        stride: buffer.stride,
                    });
                }
                _ => {}
            }
        }

        Ok(outputs)
    }
}
