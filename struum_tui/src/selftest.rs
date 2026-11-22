//! Startup self-tests and backend choice, mirroring `struum_worker`.

use std::time::Duration;

use struum_cpu::selftest;
use struum_kernel::Kernel;
use struum_node::Node;
use struum_types::network::NodeBackend;
use struum_types::{Buffer, StruumError};

const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(30);

/// Runs the built-in WebAssembly task through the local scheduler.
pub async fn cpu<const N: usize>(node: &mut Node<N>) -> Result<(), StruumError> {
    let n = 1000;
    let out = tokio::time::timeout(SELF_TEST_TIMEOUT, node.execute_local(selftest::task(n)))
        .await
        .map_err(|_| StruumError::TaskError("CPU self-test timed out".to_string()))??
        .into_cpu_task()
        .ok_or_else(|| StruumError::TaskError("CPU self-test returned GPU work".to_string()))?;
    let y = out
        .get_buffer("y")
        .ok_or_else(|| StruumError::NotFound("CPU self-test buffer 'y' missing".to_string()))?
        .as_slice::<f32>();
    if y != selftest::expected(n).as_slice() {
        return Err(StruumError::TaskError("CPU self-test produced wrong output".to_string()));
    }
    Ok(())
}

/// Runs a tiny kernel (y = 2x) through the local scheduler and checks the output.
pub async fn gpu<const N: usize>(node: &mut Node<N>) -> Result<(), StruumError> {
    let x: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let mut kernel = Kernel::new("void compute(uint i) { y[i] = 2.0 * x[i]; }", "compute");
    kernel.add_buffer::<f32>(Buffer::new("x", &x));
    kernel.add_buffer::<f32>(Buffer::empty::<f32>("y", x.len()));
    kernel.set_work_buffer("x");
    kernel.pack()?;

    let result = tokio::time::timeout(SELF_TEST_TIMEOUT, node.execute_local(kernel))
        .await
        .map_err(|_| StruumError::GlContextCreationError("GPU self-test timed out".to_string()))??
        .into_kernel()
        .ok_or_else(|| StruumError::TaskError("GPU self-test returned CPU work".to_string()))?;
    let y = result
        .get_buffer("y")
        .ok_or_else(|| StruumError::NotFound("GPU self-test buffer 'y' missing".to_string()))?
        .as_slice::<f32>();
    if y.iter().zip(x.iter()).any(|(got, v)| (got - 2.0 * v).abs() > 1e-4) {
        return Err(StruumError::ShaderCreationError("GPU self-test produced wrong output".to_string()));
    }
    Ok(())
}

/// Checks the CPU engine, then picks the backend to advertise:
/// `requested` = `Some("cpu")` / `Some("opengl")` forces one, `None` tries the GPU.
pub async fn choose_backend<const N: usize>(
    node: &mut Node<N>,
    requested: Option<&str>,
) -> Result<NodeBackend, StruumError> {
    cpu(node).await?;
    log::info!("CPU (WebAssembly) self-test passed");
    let backend = match requested {
        Some("cpu") => NodeBackend::Cpu,
        Some("opengl") => {
            gpu(node).await?;
            NodeBackend::OpenGL
        }
        Some(other) => {
            return Err(StruumError::ParserError(format!("unknown backend '{other}' (cpu|opengl)")));
        }
        None => match gpu(node).await {
            Ok(()) => NodeBackend::OpenGL,
            Err(e) => {
                log::warn!("GPU unavailable ({e}); serving CPU tasks only");
                NodeBackend::Cpu
            }
        },
    };
    log::info!("Advertising backend {backend:?} (CPU tasks always accepted)");
    Ok(backend)
}
