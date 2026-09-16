use std::fmt;

#[derive(Debug)]
pub enum StruumError {
    GlContextCreationError(String),
    ShaderCreationError(String),
    BufferWriteError(String),
    NotFound(String),
    NetworkConnectionError(String),
    SerializationError(String),
    ParserError(String),
}

impl fmt::Display for StruumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GlContextCreationError(msg) => write!(f, "GL Context Creation Error: {}", msg),
            Self::ShaderCreationError(msg) => write!(f, "Shader Creation Error: {}", msg),
            Self::BufferWriteError(msg) => write!(f, "Buffer Write Error: {}", msg),
            Self::NotFound(msg) => write!(f, "Not Found Error: {}", msg),
            Self::NetworkConnectionError(msg) => write!(f, "Network Connection Error: {}", msg),
            Self::SerializationError(msg) => write!(f, "Serialization Error: {}", msg),
            Self::ParserError(msg) => write!(f, "Parser Error: {}", msg),
        }
    }
}

impl std::error::Error for StruumError {}
