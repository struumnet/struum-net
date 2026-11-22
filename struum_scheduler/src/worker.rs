use std::sync::{Arc, OnceLock};
use tokio::sync::{Mutex, Notify, mpsc::Sender};

use struum_cpu::CpuExecutor;
use struum_kernel::Kernel;
use struum_opengl::OpenglBackend;
use struum_types::StruumError;

use crate::job::JobId;
use crate::state::SchedulerState;
use crate::work::Work;

static GL_EXEC_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// One wasm engine per process: compiled modules are cached and shared.
static CPU_EXECUTOR: OnceLock<Result<Arc<CpuExecutor>, String>> = OnceLock::new();

fn cpu_executor() -> Result<Arc<CpuExecutor>, StruumError> {
    CPU_EXECUTOR
        .get_or_init(|| CpuExecutor::new().map(Arc::new).map_err(|e| e.to_string()))
        .clone()
        .map_err(StruumError::TaskError)
}

fn run_gpu(mut kernel: Kernel) -> Result<Kernel, StruumError> {
    let _gl_lock = GL_EXEC_MUTEX.lock().unwrap();
    let backend = OpenglBackend::from_kernel(&kernel)?;
    backend.execute();
    backend.sync_to_kernel(&mut kernel)?;
    Ok(kernel)
}

async fn run(work: Work) -> Result<Work, StruumError> {
    match work {
        Work::Gpu(kernel) => run_gpu(kernel).map(Work::Gpu),
        Work::Cpu(task) => {
            let executor = cpu_executor()?;
            tokio::task::spawn_blocking(move || executor.execute(task))
                .await
                .map_err(|e| StruumError::TaskError(format!("CPU task panicked: {e}")))?
                .map(Work::Cpu)
        }
    }
}

pub(crate) async fn worker(state: Arc<Mutex<SchedulerState>>, tx: Sender<JobId>, work_ready: Arc<Notify>) {
    loop {
        let job = {
            let mut state = state.lock().await;
            state.queue.pop_front()
        };

        let Some(job) = job else {
            // notify_one() stores a permit if nobody is waiting, so no wakeup is lost.
            work_ready.notified().await;
            continue;
        };

        let id = job.id.clone();
        log::debug!("Worker starting job {id}");

        let result = run(job.work).await.map_err(|err| {
            log::error!("Worker failed job {id}: {err}");
            err.to_string()
        });

        log::debug!("Worker finished job {id}");
        if let Some(reply) = job.reply {
            // The caller may have given up (e.g. timed out); nothing else wants the result.
            let _ = reply.send(result);
            continue;
        }

        {
            let mut state = state.lock().await;
            state.results.insert(id.clone(), result);
        }

        // Always signal completion, even on failure, so callers never hang.
        if tx.send(id).await.is_err() {
            break;
        }
    }
}
