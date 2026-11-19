use std::collections::{HashMap, VecDeque};

use crate::job::{Job, JobId};
use crate::work::Work;

pub(crate) struct SchedulerState {
    pub(crate) queue: VecDeque<Job>,
    /// Finished jobs: the resulting work, or the error message.
    pub(crate) results: HashMap<JobId, Result<Work, String>>,
}
