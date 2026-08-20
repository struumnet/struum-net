mod gl_context;
mod shader;

use gl_context::GlContext;
use shader::Shader;

use struum_types::{
    StruumError,
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

    pub fn execute(&self, shader: &str, buffers: &Vec<Buffer>) -> Result<Vec<Buffer>, StruumError> {
        let shader = Shader::new(shader)?;
        println!("{:#?}", shader);
        Ok(buffers.to_vec())
    }
}
