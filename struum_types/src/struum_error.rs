#[derive(Debug)]
pub enum StruumError {
    GlContextCreationError(String),
    ShaderCreationError(String),
    BufferWriteError(String),
    NotFound(String),
}
