#[derive(Debug)]
pub enum StruumError {
    GlContextCreationError(String),
    ShaderCreationError(String),
    BufferWriteError(String),
    NotFound(String),
<<<<<<< HEAD
    NetworkConnectionError(String),
=======
>>>>>>> 7178db7 (fix: + Made shader compilation only once)
}
