use std::sync::Arc;
use tokio::sync::{mpsc::Sender, Mutex};
use std::collections::{HashMap, VecDeque};

use struum_kernel::Kernel;

use crate::state::SchedulerState;
use crate::job::{JobId, Job};
use crate::worker::worker;

#[derive(Clone)]
pub struct JobScheduler {
    state: Arc<Mutex<SchedulerState>>,
}

impl JobScheduler {
    pub fn new(tx: Sender<JobId>, workers: u32) -> Self {
        let state = Arc::new(
            Mutex::new(SchedulerState {
                queue: VecDeque::new(),
                results: HashMap::new(),
            })
        );

        // Spawn the number of workers
        for _ in 0..workers {
            let state = state.clone();
            let tx = tx.clone();

            tokio::spawn(async move {
                worker(state, tx).await;
            });
        }

        Self {
            state,
        }
    }

    pub async fn add_job(&self, kernel: Kernel) -> JobId {
        let job = Job::new(kernel);
        let id = job.id.clone();

        let mut state = self.state.lock().await;
        state.queue.push_back(job);

        id
    }

    pub async fn get_result_of_job(&self, id: &JobId) -> Option<Kernel> {
        let mut state = self.state.lock().await;
        state.results.remove(id)
    }
}
