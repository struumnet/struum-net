use std::ffi::{c_void, CString};
use gl::types::GLuint;
use struum_types::StruumError;

#[derive(Debug)]
pub(crate) struct Shader {
    id: GLuint,
}

impl Shader {
    pub(crate) fn new(src: &str) -> Result<Self, StruumError> {
        let shader = unsafe {
            gl::CreateShader(gl::COMPUTE_SHADER)
        };

        Self::compile_shader(shader, src)?;
        let program = Self::create_program(shader)?;

        Ok(Self {
            id: program,
        })
    }

    fn compile_shader(shader: GLuint, src: &str) -> Result<(), StruumError> {
        let source = CString::new(src)
            .map_err(|e| StruumError::ShaderCreationError(e.to_string()))?;

        unsafe {
            // Compile
            gl::ShaderSource(
                shader,
                1,
                &source.as_ptr(),
                std::ptr::null(),
            );
            gl::CompileShader(shader);

            // Check compilation error
            let mut success = 0;

            gl::GetShaderiv(
                shader,
                gl::COMPILE_STATUS,
                &mut success,
            );

            // Compilation failed
            if success == 0 {
                let mut len = 0;

                gl::GetShaderiv(
                    shader,
                    gl::INFO_LOG_LENGTH,
                    &mut len,
                );

                let mut buffer = vec![0u8; len as usize];

                gl::GetShaderInfoLog(
                    shader,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut i8,
                );

                let log = String::from_utf8_lossy(&buffer);

                return Err(StruumError::ShaderCreationError(
                    format!("Compute shader compilation failed:\n{}", log)
                ));
            }
        }

        Ok(())
    }

    fn create_program(shader: GLuint) -> Result<GLuint, StruumError> {
        unsafe {
            // Create program
            let program = gl::CreateProgram();

            gl::AttachShader(program, shader);
            gl::LinkProgram(program);

            gl::DeleteShader(shader);

            // Check linking
            let mut success = 0;

            gl::GetProgramiv(
                program,
                gl::LINK_STATUS,
                &mut success,
            );

            // Link failed
            if success == 0 {
                let mut len = 0;

                gl::GetProgramiv(
                    program,
                    gl::INFO_LOG_LENGTH,
                    &mut len,
                );

                let mut buffer = vec![0u8; len as usize];

                gl::GetProgramInfoLog(
                    program,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut i8,
                );

                let log = String::from_utf8_lossy(&buffer);

                return Err(StruumError::ShaderCreationError(
                    format!("Program linking failed:\n{}", log)
                ));
            }

            Ok(program)
        }
    }
}
