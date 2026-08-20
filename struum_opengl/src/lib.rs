mod gl_context;
use gl_context::GlContext;

use struum_types::StruumError;
use struum_types::Buffer;

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

    pub fn execute(shader: String, buffers: Vec<Buffer>) -> Vec<Buffer> {
        buffers
    }
}
