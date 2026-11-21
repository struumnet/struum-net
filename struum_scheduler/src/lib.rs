//! Local job execution for a node.
//!
//! A [`JobScheduler`] runs a pool of worker tasks that pull jobs from a queue. A job is a
//! [`Work`]: either a GPU [`Kernel`](struum_kernel::Kernel), run on the
//! [OpenGL backend](struum_opengl::OpenglBackend), or a CPU
//! [`CpuTask`](struum_cpu::CpuTask), run on a shared
//! [`CpuExecutor`](struum_cpu::CpuExecutor). Finished job ids are sent on the channel passed
//! to [`JobScheduler::new`], and results are collected with
//! [`JobScheduler::get_work_result`].
//!
//! [`Work`] is also the unit the node sends over the network: it can be serialized
//! ([`Work::to_bytes`]), split into shards and merged back
//! ([`Shardable`](struum_types::Shardable)).

mod job;
mod scheduler;
mod state;
mod work;
mod worker;

pub use job::JobId;
pub use scheduler::JobScheduler;
pub use work::Work;
