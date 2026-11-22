use std::sync::Arc;
use tokio::sync::{mpsc::Sender, Mutex, Notify};
use std::collections::{HashMap, VecDeque};

use struum_kernel::Kernel;
use struum_types::StruumError;

use crate::state::SchedulerState;
use crate::job::{JobId, Job};
use crate::work::Work;
use crate::worker::worker;

/// Queue of local jobs served by a pool of worker tasks.
///
/// Cloning shares the same queue and workers.
#[derive(Clone)]
pub struct JobScheduler {
    state: Arc<Mutex<SchedulerState>>,
    /// Wakes idle workers when a job is queued.
    work_ready: Arc<Notify>,
}

impl JobScheduler {
    /// Spawns `workers` worker tasks on the current tokio runtime. Each finished job's id
    /// is sent on `tx`, after which its result can be taken with
    /// [`JobScheduler::get_work_result`].
    pub fn new(tx: Sender<JobId>, workers: u32) -> Self {
        let state = Arc::new(
            Mutex::new(SchedulerState {
                queue: VecDeque::new(),
                results: HashMap::new(),
            })
        );

        let work_ready = Arc::new(Notify::new());

        // Spawn the number of workers
        for _ in 0..workers {
            let state = state.clone();
            let tx = tx.clone();
            let work_ready = work_ready.clone();

            tokio::spawn(async move {
                worker(state, tx, work_ready).await;
            });
        }

        Self {
            state,
            work_ready,
        }
    }

    /// Queues a GPU kernel (shorthand for `add_work(Work::Gpu(kernel))`).
    pub async fn add_job(&self, kernel: Kernel) -> JobId {
        self.add_work(Work::Gpu(kernel)).await
    }

    /// Queues any kind of work.
    pub async fn add_work(&self, work: Work) -> JobId {
        let job = Job::new(work);
        let id = job.id.clone();

        self.state.lock().await.queue.push_back(job);
        self.work_ready.notify_one();

        id
    }

    /// Queues `work` and waits for its result. Unlike [`JobScheduler::add_work`],
    /// nothing is sent on the completion channel, so any number of callers can
    /// run work at once from cloned handles.
    pub async fn run(&self, work: Work) -> Result<Work, StruumError> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let mut job = Job::new(work);
        job.reply = Some(tx);
        self.state.lock().await.queue.push_back(job);
        self.work_ready.notify_one();
        rx.await
            .map_err(|_| StruumError::TaskError("local scheduler stopped".to_string()))?
            .map_err(StruumError::TaskError)
    }

    /// Result of a finished GPU job; `None` if it failed, is unknown or was CPU work.
    pub async fn get_result_of_job(&self, id: &JobId) -> Option<Kernel> {
        match self.get_work_result(id).await {
            Some(Ok(work)) => work.into_kernel(),
            Some(Err(e)) => {
                log::error!("Job {id} failed: {e}");
                None
            }
            None => None,
        }
    }

    /// Takes the result of a finished job: `Some(Ok(work))`, `Some(Err(message))`, or `None`
    /// if the id is unknown, unfinished or already taken.
    pub async fn get_work_result(&self, id: &JobId) -> Option<Result<Work, String>> {
        let mut state = self.state.lock().await;
        state.results.remove(id)
    }
}
