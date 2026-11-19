use std::sync::Arc;
use tokio::sync::{mpsc::Sender, Mutex, Notify};
use std::collections::{HashMap, VecDeque};

use struum_kernel::Kernel;

use crate::state::SchedulerState;
use crate::job::{JobId, Job};
use crate::work::Work;
use crate::worker::worker;

#[derive(Clone)]
pub struct JobScheduler {
    state: Arc<Mutex<SchedulerState>>,
    /// Wakes idle workers when a job is queued.
    work_ready: Arc<Notify>,
}

impl JobScheduler {
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

    /// Result of a finished job: `Some(Ok(work))`, `Some(Err(message))`, or `None` if unknown.
    pub async fn get_work_result(&self, id: &JobId) -> Option<Result<Work, String>> {
        let mut state = self.state.lock().await;
        state.results.remove(id)
    }
}
