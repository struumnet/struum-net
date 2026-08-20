#[derive(Debug)]
pub enum StruumError {
    GlContextCreationError(String),
    ShaderCreationError(String),
}
