use std::sync::Arc;
use tokio::sync::{mpsc::Sender, Mutex};

use struum_opengl::OpenglBackend;

use crate::state::SchedulerState;
use crate::job::JobId;

static GL_EXEC_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) async fn worker(state: Arc<Mutex<SchedulerState>>, tx: Sender<JobId>) {
    // T_T ???
    loop {
        let job = {
            let mut state = state.lock().await;
            state.queue.pop_front()
        };

        let Some(mut job) = job else {
            tokio::time::sleep(
                std::time::Duration::from_millis(10)
            ).await;

            continue;
        };

        let id = job.id.clone();

        log::debug!("Worker starting job {id}");

        {
            let _gl_lock = GL_EXEC_MUTEX.lock().unwrap();
            let backend = match OpenglBackend::from_kernel(&job.kernel) {
                Ok(backend) => backend,
                Err(err) => {
                    log::error!("Worker failed to create backend for job {id}: {err:?}");
                    continue;
                }
            };
            backend.execute();
            backend.sync_to_kernel(&mut job.kernel).unwrap(); //TODO(slok): Remove unwrap
        };

        log::debug!("Worker finished job {id}");

        {
            let mut state = state.lock().await;
            state.results.insert(id.clone(), job.kernel);
        }

        if tx.send(id).await.is_err() {
            break;
        }
    }
}

