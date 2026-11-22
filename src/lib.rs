//! # struum-net
//!
//! A distributed GPU/CPU compute network. Devices on a LAN find each other through a
//! relay, then a requester splits a job into shards and streams them to workers over TCP.
//! Workers run GLSL compute shaders (OpenGL / OpenGL ES) or WebAssembly CPU tasks, and the
//! requester merges the results in order.
//!
//! This package only ties the workspace together for the examples. The functionality
//! lives in the member crates:
//!
//! | Layer | Crate | What it provides |
//! |---|---|---|
//! | Discovery | [`struum_relay`] | [`RelayServer`](struum_relay::RelayServer): registers nodes and introduces them to each other over UDP |
//! | Node | [`struum_node`] | [`Node`](struum_node::Node): peer discovery, TCP connections, [`run_sharded`](struum_node::Node::run_sharded) |
//! | | [`struum_coordinator`] | [`Coordinator`](struum_coordinator::Coordinator): a node acting as a network coordinator |
//! | Transport | [`struum_network`] | [`NetworkLayer`](struum_network::layer::NetworkLayer): UDP datagrams and length-prefixed TCP framing |
//! | Local execution | [`struum_scheduler`] | [`JobScheduler`](struum_scheduler::JobScheduler) and [`Work`](struum_scheduler::Work): runs jobs on worker threads |
//! | GPU backend | [`struum_kernel`], [`struum_opengl`] | [`Kernel`](struum_kernel::Kernel) (GLSL + buffers) and [`OpenglBackend`](struum_opengl::OpenglBackend) |
//! | | [`struum_macros`] | [`#[gpu_type]`](struum_macros::gpu_type): Rust structs usable as GLSL structs |
//! | CPU backend | [`struum_cpu`] | [`CpuTask`](struum_cpu::CpuTask) and [`CpuExecutor`](struum_cpu::CpuExecutor) (wasmtime) |
//! | | `struum_guest` | Guest side of CPU tasks, compiled to `wasm32-unknown-unknown` |
//! | Shared types | [`struum_types`] | [`Buffer`](struum_types::Buffer), sharding primitives, packets, [`StruumError`](struum_types::StruumError) |
//!
//! The standalone worker binary is `struum_worker` (desktop or Android).
//!
//! ## A job end to end
//!
//! 1. Start a relay (`cargo run -p struum_relay`) and one or more workers
//!    (`cargo run -p struum_worker`).
//! 2. The requester creates a [`Node`](struum_node::Node), discovers the workers through
//!    the relay with [`discover_peers`](struum_node::Node::discover_peers) and connects to
//!    them with [`connect_to_peers`](struum_node::Node::connect_to_peers).
//! 3. It builds a [`CpuTask`](struum_cpu::CpuTask) or a [`Kernel`](struum_kernel::Kernel)
//!    and calls [`run_sharded`](struum_node::Node::run_sharded). Buffers are split or
//!    broadcast per shard according to their [`ShardMode`](struum_types::ShardMode).
//!
//! See the `examples/` directory (`sharded_cpu`, `sharded_gpu`, `mnist_train`, `render`,
//! `bench`) and the README for setup.
