use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;
use struum_types::StruumError;

#[derive(Debug)]
pub(crate) struct Shader {
    id: u32,
}

impl Shader {
    pub(crate) fn new(src: &str) -> Result<Self, StruumError> {
        let shader = unsafe { gl::CreateShader(gl::COMPUTE_SHADER) };

        Self::compile_shader(shader, src)?;
        let program = Self::create_program(shader)?;

        Ok(Self { id: program })
    }

    pub(crate) fn dispatch_and_wait(&self, x: u32, y: u32, z: u32) {
        unsafe {
            gl::UseProgram(self.id);

            gl::DispatchCompute(x, y, z);

            // Make shader writes visible to later buffer reads/maps
            gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT | gl::BUFFER_UPDATE_BARRIER_BIT);
        }
    }

    fn compile_shader(shader: u32, src: &str) -> Result<(), StruumError> {
        let source =
            CString::new(src).map_err(|e| StruumError::ShaderCreationError(e.to_string()))?;

        unsafe {
            // Compile
            gl::ShaderSource(shader, 1, &source.as_ptr(), std::ptr::null());
            gl::CompileShader(shader);

            // Check compilation error
            let mut success = 0;

            gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut success);

            // Compilation failed
            if success == 0 {
                let mut len = 0;

                gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);

                let mut buffer = vec![0u8; len as usize];

                gl::GetShaderInfoLog(
                    shader,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut gl::types::GLchar,
                );

                let log = String::from_utf8_lossy(&buffer);

                return Err(StruumError::ShaderCreationError(format!(
                    "Compute shader compilation failed:\n{}",
                    log
                )));
            }
        }

        Ok(())
    }

    fn create_program(shader: u32) -> Result<u32, StruumError> {
        unsafe {
            // Create program
            let program = gl::CreateProgram();

            gl::AttachShader(program, shader);
            gl::LinkProgram(program);

            gl::DeleteShader(shader);

            // Check linking
            let mut success = 0;

            gl::GetProgramiv(program, gl::LINK_STATUS, &mut success);

            // Link failed
            if success == 0 {
                let mut len = 0;

                gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);

                let mut buffer = vec![0u8; len as usize];

                gl::GetProgramInfoLog(
                    program,
                    len,
                    std::ptr::null_mut(),
                    buffer.as_mut_ptr() as *mut gl::types::GLchar,
                );

                let log = String::from_utf8_lossy(&buffer);

                return Err(StruumError::ShaderCreationError(format!(
                    "Program linking failed:\n{}",
                    log
                )));
            }

            Ok(program)
        }
    }
}

/// Compiled programs of the GL context on this thread, by adapted source.
struct ProgramCache(HashMap<String, Rc<Shader>>);

/// Programs still cached at thread exit are left to the context teardown:
/// deleting them here could run after the thread's context is gone.
impl Drop for ProgramCache {
    fn drop(&mut self) {
        for (_, shader) in self.0.drain() {
            std::mem::forget(shader);
        }
    }
}

const PROGRAM_CACHE_LIMIT: usize = 64;

thread_local! {
    static PROGRAMS: RefCell<ProgramCache> = RefCell::new(ProgramCache(HashMap::new()));
}

impl Shader {
    /// Returns the program for `src`, compiling it only the first time it is seen
    /// on this thread (sharded GPU jobs send the same kernel over and over).
    pub(crate) fn cached(src: &str) -> Result<Rc<Shader>, StruumError> {
        if let Some(hit) = PROGRAMS.with(|c| c.borrow().0.get(src).cloned()) {
            return Ok(hit);
        }
        let shader = Rc::new(Shader::new(src)?);
        PROGRAMS.with(|c| {
            let mut cache = c.borrow_mut();
            if cache.0.len() >= PROGRAM_CACHE_LIMIT {
                // The context is current here, so dropping deletes the programs properly.
                cache.0.clear();
            }
            cache.0.insert(src.to_string(), shader.clone());
        });
        Ok(shader)
    }

    /// Number of programs cached on this thread.
    #[cfg(test)]
    pub(crate) fn cached_count() -> usize {
        PROGRAMS.with(|c| c.borrow().0.len())
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteProgram(self.id);
        }
    }
}
