//! A struum node: discovers peers, serves work, and runs sharded jobs.
//!
//! A [`Node`] is both a worker and a requester. It embeds a
//! [`NetworkLayer`](struum_network::layer::NetworkLayer) for transport and a
//! [`JobScheduler`](struum_scheduler::JobScheduler) for running work locally on CPU
//! (wasm) or GPU (OpenGL).
//!
//! - **As a worker**, it registers with a relay, accepts connections
//!   ([`Node::accept_peer`]) and runs whatever its peers send ([`PeerServer::serve`]),
//!   serving several requesters at once. [`serving`] does this over QUIC (and TCP
//!   when no token is set).
//! - **As a requester**, it discovers workers ([`Node::discover_peers`]), connects to them
//!   ([`Node::connect_to_peers`]) and runs jobs across them with [`Node::run_sharded`].
//!
//! # Sharded jobs
//!
//! [`Node::run_sharded`] splits the work range into `peers × shards_per_peer` shards.
//! Peers pull shards from a shared queue, so faster peers take more. A shard whose peer
//! times out or disconnects is reassigned; a task error (wasm trap, bad shader) fails the
//! job. See [`ShardOptions`] for tuning and [`ShardReport`] for what happened.
//!
//! ```no_run
//! use std::time::Duration;
//! use struum_node::{CpuTask, Node, ShardOptions};
//! use struum_types::{Buffer, network::NodeBackend};
//!
//! # async fn run() -> Result<(), struum_types::StruumError> {
//! let mut node = Node::<2048>::new(0, 0, NodeBackend::Cpu, 1).await?;
//! let relay = "192.168.1.10:39001".parse().unwrap();
//! let peers = node.discover_peers(relay, 2, Duration::from_secs(2)).await?;
//! node.connect_to_peers(&peers).await;
//!
//! let limits: Vec<u32> = (0..1000).collect();
//! let mut task = CpuTask::new(std::fs::read("tasks.wasm").unwrap(), "count_primes");
//! task.add_buffer(Buffer::new("limits", &limits)); // split: one element per work item
//! task.add_buffer(Buffer::empty::<u32>("counts", limits.len()));
//! task.set_work_buffer("limits");
//!
//! let result = node.run_sharded(task, &ShardOptions::default()).await?;
//! let counts = result.into_cpu_task().unwrap();
//! # Ok(()) }
//! ```

pub mod dispatch;
pub mod node;
pub mod serving;
pub mod stats;

pub use dispatch::{ShardOptions, ShardReport};
pub use node::{Coordinator, Node, PeerServer};
pub use stats::{NodeStats, PeerTally, StatsSnapshot, Tally};
pub use struum_cpu::CpuTask;
pub use struum_scheduler::Work;
