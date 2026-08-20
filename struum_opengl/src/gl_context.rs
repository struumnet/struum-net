use std::ffi::c_void;
use glfw::Context;
use struum_types::StruumError;

pub(crate) struct GlContext {
    _glfw: glfw::Glfw,
    _window: glfw::PWindow,
}

impl GlContext {
    pub(crate) fn new() -> Result<Self, StruumError> {
        let mut glfw = glfw::init(glfw::fail_on_errors)
            .map_err(|e| StruumError::GlContextCreationError(e.to_string()))?;

        glfw.window_hint(glfw::WindowHint::ContextVersion(4, 3));
        glfw.window_hint(glfw::WindowHint::Visible(false));

        let (mut window, _) = glfw
            .create_window(
                640, 480,
                "struum-opengl",
                glfw::WindowMode::Windowed
            )
            .ok_or(StruumError::GlContextCreationError(
                "Failed to create glfw window".to_string()
            ))?;
        window.make_current();

        gl::load_with(|name| {
            window
                .get_proc_address(name)
                .map_or(std::ptr::null(), |p| p as *const c_void)
        });

        Ok(Self {
            _glfw: glfw,
            _window: window,
        })
    }
}
