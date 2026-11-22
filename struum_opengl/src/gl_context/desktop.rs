use glfw::Context;
use std::cell::RefCell;
use std::ffi::c_void;
use std::sync::{Mutex, Once};
use struum_types::StruumError;

thread_local! {
    static THREAD_GL_CONTEXT: RefCell<Option<GlContext>> = const { RefCell::new(None) };
}

/// Serializes every glfw call that touches global state (init, window
/// creation and destruction): glfw is not thread-safe, but each worker thread
/// owns its own hidden window and GL context.
static GLFW_LOCK: Mutex<()> = Mutex::new(());

/// Configures the process environment once, before glfw is first initialized.
static ENV_SETUP: Once = Once::new();

/// Hidden glfw window with an OpenGL 4.3 core context, one per thread that runs
/// GPU work (see [`GlContext::ensure_current`]).
///
/// glfw is initialized once per process and never terminated: `glfw::Glfw`
/// calls `glfwTerminate` when its last handle drops, which would destroy the
/// windows and contexts other threads are still using. One handle is leaked
/// on first init so that never happens.
pub(crate) struct GlContext {
    window: Option<glfw::PWindow>,
    _glfw: glfw::Glfw,
}

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
        let _guard = GLFW_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        ENV_SETUP.call_once(|| {
            // On Linux X11/Mesa with certain GPU drivers, DRI3 negotiation for
            // invisible offscreen windows can fail.
            if std::env::var_os("LIBGL_DRI3_DISABLE").is_none() {
                // SAFETY: done once, under GLFW_LOCK, before any GL driver code runs.
                unsafe {
                    std::env::set_var("LIBGL_DRI3_DISABLE", "1");
                }
            }
        });

        let mut glfw = glfw::init(glfw::log_errors)
            .map_err(|e| StruumError::GlContextCreationError(e.to_string()))?;
        static KEEP_GLFW_ALIVE: Once = Once::new();
        KEEP_GLFW_ALIVE.call_once(|| std::mem::forget(glfw.clone()));

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
            window: Some(window),
            _glfw: glfw,
        })
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        // Destroy the window (and its context) under the lock, before the glfw handle.
        let _guard = GLFW_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        drop(self.window.take());
    }
}
