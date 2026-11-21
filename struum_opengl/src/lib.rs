//! OpenGL compute backend for GPU [`Kernel`](struum_kernel::Kernel)s.
//!
//! [`OpenglBackend`] compiles a kernel's GLSL, uploads its buffers as shader storage
//! buffers, dispatches it and reads the results back.
//!
//! # Contexts
//!
//! A GL context is created lazily, once per thread:
//! - **Desktop (default)**: a hidden glfw window with an OpenGL 4.3 core context.
//! - **Android, or the `egl` feature**: a headless EGL context with OpenGL ES 3.1.
//!   Desktop GLSL is rewritten to GLSL ES on the fly; kernels using `double` are rejected.
//!
//! Compiled programs are cached per thread by source, so a kernel sent as many shards
//! is compiled only once.

mod gl_backend;
mod gl_buffer;
mod gl_context;
mod glsl;
mod shader;

pub use gl_backend::OpenglBackend;
