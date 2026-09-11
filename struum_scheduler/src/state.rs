use std::collections::{HashMap, VecDeque};
use struum_kernel::Kernel;

use crate::job::{JobId, Job};

pub(crate) struct SchedulerState {
    pub(crate) queue: VecDeque<Job>,
    pub(crate) results: HashMap<JobId, Kernel>,
}

