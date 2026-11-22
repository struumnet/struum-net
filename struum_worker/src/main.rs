//! Standalone struum worker.
//!
//! Registers with a relay, accepts requesters over QUIC (and TCP when no token
//! is set, see `struum_node::serving`), and runs the
//! work they send: GPU kernels (OpenGL) and CPU tasks (WebAssembly), including
//! shards of larger jobs. Builds for desktop (glfw + OpenGL 4.3) and for Android
//! (headless EGL + OpenGL ES 3.1), see README "Android worker".
//!
//! Configuration (env vars):
//! - `RELAY_ADDR`    relay UDP address (default `127.0.0.1:39001`)
//! - `NODE_ID`       numeric node id (default: random)
//! - `NODE_UDP_PORT` UDP port (default 34255)
//! - `NODE_TCP_PORT` TCP port (default 34265)
//! - `NODE_QUIC_PORT` QUIC (UDP) port (default 34275)
//! - `STRUUM_TOKEN`  shared secret requesters must prove they know; with it set,
//!                   TCP is off and only QUIC is served
//! - `WORKER_BACKEND` `opengl` or `cpu` (default: `opengl` if the GPU self-test passes)
//! - `RUST_LOG`      log filter (default `info`)

use std::net::SocketAddr;
use std::time::Duration;

use struum_cpu::selftest;
use struum_kernel::Kernel;
use struum_node::node::Node;
use struum_node::serving;
use struum_types::network::{NodeBackend, UID};
use struum_types::{Buffer, StruumError};

const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(30);

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let udp_port: u16 = env_or("NODE_UDP_PORT", 34255);
    let tcp_port: u16 = env_or("NODE_TCP_PORT", 34265);
    let quic_port: u16 = env_or("NODE_QUIC_PORT", 34275);
    let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
        .parse()
        .map_err(|e| StruumError::ParserError(format!("Invalid RELAY_ADDR: {e}")))?;
    let node_id = std::env::var("NODE_ID")
        .ok()
        .and_then(|p| p.parse().ok())
        .map(UID::new)
        .unwrap_or_else(UID::generate);

    let mut node = Node::<2048>::with_id(node_id, udp_port, tcp_port, NodeBackend::OpenGL, 1).await?;

    log::info!(
        "struum_worker {} on {} ({}): UDP {}:{} / TCP {}:{}, relay {}",
        node.id,
        std::env::consts::OS,
        std::env::consts::ARCH,
        node.net.ip,
        node.net.udp_port,
        node.net.ip,
        node.net.tcp_port,
        relay_addr
    );

    // Every worker runs CPU tasks; check the wasm engine works before registering.
    cpu_self_test(&mut node).await?;

    // Advertise OpenGL only if the GPU can actually run compute shaders.
    node.backend = match std::env::var("WORKER_BACKEND").ok().as_deref() {
        Some("cpu") => NodeBackend::Cpu,
        Some("opengl") => {
            gpu_self_test(&mut node).await?;
            NodeBackend::OpenGL
        }
        _ => match gpu_self_test(&mut node).await {
            Ok(()) => NodeBackend::OpenGL,
            Err(e) => {
                log::warn!("GPU unavailable ({e}); serving CPU tasks only");
                NodeBackend::Cpu
            }
        },
    };
    log::info!("Advertising backend {:?} (CPU tasks always accepted)", node.backend);

    let token = struum_network::quic::token_from_env();
    let listening = serving::prepare(&mut node, quic_port, &token)?;
    log::info!(
        "Serving QUIC on UDP {}, TCP {}, auth {}",
        listening.quic_port,
        listening.tcp_port.map_or("off".to_string(), |p| p.to_string()),
        if listening.authenticated { "token" } else { "open" }
    );

    // Registers the QUIC port (and TCP only if it's on) with the relay.
    let _peer = node.long_poll_relay(relay_addr).await?;

    // Each requester gets its own task; they share this node's workers.
    serving::spawn(node, listening);
    std::future::pending::<()>().await;
    Ok(())
}

/// Runs the built-in WebAssembly task through the local scheduler.
async fn cpu_self_test(node: &mut Node<2048>) -> Result<(), StruumError> {
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
    log::info!("CPU (WebAssembly) self-test passed ({} elements)", n);
    Ok(())
}

/// Runs a tiny kernel (y = 2x) through the local scheduler and checks the output.
async fn gpu_self_test(node: &mut Node<2048>) -> Result<(), StruumError> {
    let x: Vec<f32> = (0..300).map(|i| i as f32).collect();

    let mut kernel = Kernel::new("void compute(uint i) { y[i] = 2.0 * x[i]; }", "compute");
    kernel.add_buffer::<f32>(Buffer::new("x", &x));
    kernel.add_buffer::<f32>(Buffer::empty::<f32>("y", x.len()));
    kernel.set_work_buffer("x");
    kernel.pack()?;

    let result = tokio::time::timeout(SELF_TEST_TIMEOUT, node.execute_local(kernel))
        .await
        .map_err(|_| {
            StruumError::GlContextCreationError(
                "GPU self-test timed out (compute shaders need OpenGL 4.3 / OpenGL ES 3.1)".to_string(),
            )
        })??
        .into_kernel()
        .ok_or_else(|| StruumError::TaskError("GPU self-test returned CPU work".to_string()))?;
    let y = result
        .get_buffer("y")
        .ok_or_else(|| StruumError::NotFound("Self-test buffer 'y' missing".to_string()))?
        .as_slice::<f32>();

    if let Some((i, (got, want))) = y
        .iter()
        .zip(x.iter().map(|v| v * 2.0))
        .enumerate()
        .find(|(_, (got, want))| (**got - *want).abs() > 1e-4)
    {
        return Err(StruumError::ShaderCreationError(format!(
            "GPU self-test produced wrong output at index {i}: got {got}, want {want}"
        )));
    }

    log::info!("GPU self-test passed ({} elements)", x.len());
    Ok(())
}
