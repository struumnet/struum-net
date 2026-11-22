use uuid::Uuid;

use crate::work::Work;

/// Identifier of a local job (a UUID string).
pub type JobId = String;

pub(crate) struct Job {
    pub(crate) id: JobId,
    pub(crate) work: Work,
    /// Set by [`crate::JobScheduler::run`]: the result goes straight to the
    /// caller instead of the results map and completion channel.
    pub(crate) reply: Option<tokio::sync::oneshot::Sender<Result<Work, String>>>,
}

impl Job {
    pub(crate) fn new(work: Work) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            work,
            reply: None,
        }
    }
}
