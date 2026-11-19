//! OpenGL context providers.
//!
//! - Desktop (default): a hidden glfw window with an OpenGL 4.3 core context.
//! - Android, or desktop with the `egl` feature: a headless EGL pbuffer with an
//!   OpenGL ES 3.1 context (the minimum version with compute shaders).

#[cfg(any(target_os = "android", feature = "egl"))]
mod egl;
#[cfg(any(target_os = "android", feature = "egl"))]
pub(crate) use egl::GlContext;

#[cfg(not(any(target_os = "android", feature = "egl")))]
mod desktop;
#[cfg(not(any(target_os = "android", feature = "egl")))]
pub(crate) use desktop::GlContext;

/// Whether the active context is OpenGL ES (shaders need GLSL ES).
pub(crate) const fn is_gles() -> bool {
    cfg!(any(target_os = "android", feature = "egl"))
}
