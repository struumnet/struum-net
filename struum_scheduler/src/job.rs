use uuid::Uuid;

use crate::work::Work;

pub type JobId = String;

pub(crate) struct Job {
    pub(crate) id: JobId,
    pub(crate) work: Work,
}

impl Job {
    pub(crate) fn new(work: Work) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            work,
        }
    }
}
