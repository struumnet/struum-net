use glfw::Context;
use std::cell::RefCell;
use std::ffi::c_void;
use struum_types::StruumError;

thread_local! {
    static THREAD_GL_CONTEXT: RefCell<Option<GlContext>> = const { RefCell::new(None) };
}

pub(crate) struct GlContext {
    _glfw: glfw::Glfw,
    _window: glfw::PWindow,
}

static INIT_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

impl GlContext {
    pub(crate) fn ensure_current() -> Result<(), StruumError> {
        THREAD_GL_CONTEXT.with(|cell| {
            let mut opt = cell.borrow_mut();
            if opt.is_none() {
                let ctx = Self::new()?;
                *opt = Some(ctx);
            }
            Ok(())
        })
    }

    pub(crate) fn new() -> Result<Self, StruumError> {
        let _guard = INIT_MUTEX.lock().unwrap();

        // On Linux X11/Mesa with certain GPU drivers, DRI3 negotiation for
        // invisible offscreen windows can fail
        if std::env::var_os("LIBGL_DRI3_DISABLE").is_none() {
            unsafe {
                std::env::set_var("LIBGL_DRI3_DISABLE", "1");
            }
        }

        let mut glfw = glfw::init(glfw::log_errors)
            .map_err(|e| StruumError::GlContextCreationError(e.to_string()))?;

        glfw.window_hint(glfw::WindowHint::ContextVersion(4, 3));
        glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
        glfw.window_hint(glfw::WindowHint::OpenGlForwardCompat(true));
        glfw.window_hint(glfw::WindowHint::Visible(false));

        let (mut window, _) = glfw
            .create_window(640, 480, "struum-opengl", glfw::WindowMode::Windowed)
            .ok_or(StruumError::GlContextCreationError(
                "Failed to create glfw window".to_string(),
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
