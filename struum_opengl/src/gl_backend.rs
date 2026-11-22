use crate::gl_buffer::GlBuffer;
use crate::gl_context::GlContext;
use crate::glsl::adapt_source;
use crate::shader::Shader;

use std::collections::HashMap;
use std::rc::Rc;

use struum_types::{Buffer, StruumError};
use struum_kernel::Kernel;

/// A compiled compute shader with its buffers uploaded to the GPU, on the GL context of
/// the current thread.
///
/// Not `Send`: create, run and read it on one thread.
pub struct OpenglBackend {
    shader: Rc<Shader>,
    gl_buffers: HashMap<String, GlBuffer>,
    group_size: u32,
}

impl OpenglBackend {
    /// Compiles a complete GLSL compute `shader` and binds `buffers` at binding points
    /// 0, 1, ... in order. [`execute`](Self::execute) dispatches `group_size` work groups.
    ///
    /// # Errors
    /// If no GL context can be created or the shader fails to compile.
    pub fn new(shader: &str, buffers: &[Buffer], group_size: u32) -> Result<Self, StruumError> {
        GlContext::ensure_current()?;

        // Compile Shader
        let shader = Shader::cached(&adapt_source(shader)?)?;

        // Construct opengl buffers
        let mut gl_buffers = HashMap::new();

        for (binding, buffer) in buffers.iter().enumerate() {
            let gl_buffer = GlBuffer::new(buffer);
            gl_buffer.bind(binding as u32);
            gl_buffers.insert(gl_buffer.name.clone(), gl_buffer);
        }

        Ok(Self {
            shader,
            gl_buffers,
            group_size,
        })
    }

    /// Prepares a packed [`Kernel`] (see [`Kernel::pack`]).
    ///
    /// # Errors
    /// If the kernel is not packed, no GL context can be created or the shader fails to compile.
    pub fn from_kernel(kernel: &Kernel) -> Result<Self, StruumError> {
        GlContext::ensure_current()?;

        // Compile Shader
        let source = kernel.get_gpu_source().ok_or(StruumError::NotFound(
            "GPU Source is None, Maybe kernel is not packed using 'pack' method".to_string(),
        ))?;
        let shader = Shader::cached(&adapt_source(source)?)?;

        // Construct opengl buffers
        let mut gl_buffers = HashMap::new();

        for (binding_id, binding) in kernel.get_buffer_bindings().iter().enumerate() {
            let gl_buffer = GlBuffer::new(&binding.buffer);
            gl_buffer.bind(binding_id as u32);
            gl_buffers.insert(gl_buffer.name.clone(), gl_buffer);
        }

        let group_size = kernel
            .get_group_size()
            .ok_or(StruumError::NotFound(
                "GPU Group Size is None, Maybe kernel is not packed using 'pack' method"
                    .to_string()
            ))?;

        Ok(Self {
            shader,
            gl_buffers,
            group_size,
        })
    }

    /// Dispatches the shader and waits until its buffer writes are visible.
    pub fn execute(&self) {
        self.shader.dispatch_and_wait(self.group_size, 1, 1);
    }

    /// Overwrites buffer `buffer_name` on the GPU with `data`.
    pub fn write_buffer<T: bytemuck::Pod>(
        &mut self,
        buffer_name: &str,
        data: &[T]
    ) -> Result<(), StruumError> {
        let gl_buffer = self
            .gl_buffers
            .get_mut(buffer_name)
            .ok_or_else(|| StruumError::NotFound(buffer_name.to_string()))?;

        let data = bytemuck::cast_slice(data);
        gl_buffer.write(data)?;

        Ok(())
    }

    /// Reads buffer `buffer_name` back from the GPU.
    pub fn read_buffer<T: bytemuck::Pod>(
        &mut self,
        buffer_name: &str,
    ) -> Result<Vec<T>, StruumError> {
        let gl_buffer = self
            .gl_buffers
            .get_mut(buffer_name)
            .ok_or_else(|| StruumError::NotFound(buffer_name.to_string()))?;

        let data = gl_buffer.read()?;

        Ok(bytemuck::cast_slice(&data).to_vec())
    }

    /// Copies every GPU buffer back into the matching buffer of `kernel`.
    pub fn sync_to_kernel(&self, kernel: &mut Kernel) -> Result<(), StruumError> {
        for (_, gl_buffer) in &self.gl_buffers {
            let data = gl_buffer.read()?;

            let binding = kernel
                .get_buffer_bindings_mut()
                .iter_mut()
                .find(|binding| binding.buffer.name == gl_buffer.name)
                .ok_or_else(|| {
                    StruumError::NotFound(
                        format!("Buffer '{}' not found in kernel", gl_buffer.name)
                    )
                })?;

            binding.buffer.data = data;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOUBLE: &str = r#"#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer X { float x[]; };
void main() { x[gl_GlobalInvocationID.x] *= 2.0; }
"#;

    #[test]
    fn programs_are_compiled_once_per_thread() {
        // Own thread: a fresh GL context and an empty program cache.
        std::thread::spawn(|| {
            for round in 0..3 {
                let input = [1.0f32, 2.0, 3.0];
                let mut backend = OpenglBackend::new(DOUBLE, &[Buffer::new("x", &input)], 3)
                    .expect("GL backend");
                backend.execute();
                assert_eq!(backend.read_buffer::<f32>("x").unwrap(), vec![2.0, 4.0, 6.0], "round {round}");
                assert_eq!(Shader::cached_count(), 1);
            }
        })
        .join()
        .unwrap();
    }
}
