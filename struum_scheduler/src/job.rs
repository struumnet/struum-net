use uuid::Uuid;
use struum_kernel::Kernel;

pub(crate) type JobId = String;

pub(crate) struct Job {
    pub(crate) id: JobId,
    pub(crate) kernel: Kernel,
}

impl Job {
    pub(crate) fn new(kernel: Kernel) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            kernel,
        }
    }
}
