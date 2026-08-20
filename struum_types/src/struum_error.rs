#[derive(Debug)]
pub enum StruumError<'a> {
    NetworkConnectionError(&'a str),
    GlContextCreationError(&'a str),
    ShaderCreationError(&'a str),
}
