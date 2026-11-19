use khronos_egl as egl;
use std::cell::RefCell;
use std::ffi::c_void;
use struum_types::StruumError;

thread_local! {
    static THREAD_GL_CONTEXT: RefCell<Option<GlContext>> = const { RefCell::new(None) };
}

// On Android `khronos-egl` is built with `no-pkg-config`, so link libEGL here.
#[cfg(target_os = "android")]
#[link(name = "EGL")]
unsafe extern "C" {}

static INIT_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

// EGL_CONTEXT_CLIENT_VERSION (EGL 1.3) and EGL_CONTEXT_MINOR_VERSION_KHR.
// Raw values so we don't depend on the EGL 1.5 constant set.
const CONTEXT_CLIENT_VERSION: egl::Int = 0x3098;
const CONTEXT_MINOR_VERSION: egl::Int = 0x30FB;

/// Headless EGL context running OpenGL ES 3.1, used on Android (and on desktop
/// with the `egl` feature). Uses a 1x1 pbuffer since compute never presents.
pub(crate) struct GlContext {
    egl: egl::Instance<egl::Static>,
    display: egl::Display,
    surface: egl::Surface,
    context: egl::Context,
}

fn egl_err(step: &str, e: egl::Error) -> StruumError {
    StruumError::GlContextCreationError(format!("{step} failed: {e}"))
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
        let _guard = INIT_MUTEX.lock().unwrap();

        let egl = egl::Instance::new(egl::Static);

        #[allow(unused_unsafe)]
        let display = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }.ok_or_else(|| {
            StruumError::GlContextCreationError("eglGetDisplay returned no display".to_string())
        })?;

        let (major, minor) = egl
            .initialize(display)
            .map_err(|e| egl_err("eglInitialize", e))?;

        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| egl_err("eglBindAPI(OPENGL_ES)", e))?;

        let config_attrs = [
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::SURFACE_TYPE,
            egl::PBUFFER_BIT,
            egl::NONE,
        ];
        let config = egl
            .choose_first_config(display, &config_attrs)
            .map_err(|e| egl_err("eglChooseConfig", e))?
            .ok_or_else(|| {
                StruumError::GlContextCreationError(
                    "No EGL config supports OpenGL ES 3 with pbuffers".to_string(),
                )
            })?;

        let context_attrs = [CONTEXT_CLIENT_VERSION, 3, CONTEXT_MINOR_VERSION, 1, egl::NONE];
        let context = egl
            .create_context(display, config, None, &context_attrs)
            .map_err(|e| egl_err("eglCreateContext(ES 3.1)", e))?;

        let surface_attrs = [egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE];
        let surface = egl
            .create_pbuffer_surface(display, config, &surface_attrs)
            .map_err(|e| egl_err("eglCreatePbufferSurface", e))?;

        egl.make_current(display, Some(surface), Some(surface), Some(context))
            .map_err(|e| egl_err("eglMakeCurrent", e))?;

        gl::load_with(|name| {
            egl.get_proc_address(name)
                .map_or(std::ptr::null(), |p| p as *const c_void)
        });

        if !gl::DispatchCompute::is_loaded() {
            return Err(StruumError::GlContextCreationError(
                "glDispatchCompute unavailable: device lacks OpenGL ES 3.1 compute".to_string(),
            ));
        }

        log::info!("Created headless EGL {major}.{minor} context with OpenGL ES 3.1");

        Ok(Self {
            egl,
            display,
            surface,
            context,
        })
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        let _ = self.egl.make_current(self.display, None, None, None);
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.context);
    }
}
