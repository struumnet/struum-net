//! Sharded execution of work across connected peer nodes.
//!
//! The job's work range is split into `peers * shards_per_peer` shards, which
//! sit in a shared queue. One task per peer pulls shards, sends them as `WORK`
//! packets and waits for the `RESULT`, so faster peers take more shards.
//!
//! - Timeout / connection error: the shard goes back on the queue for another
//!   peer (up to `max_attempts` tries) and that peer is retired.
//! - `WORKERROR` (shader error, wasm trap, ...): deterministic, so the whole job fails.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify};
use tokio::task::JoinSet;

use struum_network::layer::PeerConnection;
use struum_scheduler::Work;
use struum_types::network::{NodeBackend, NodeDetails, TcpPacket, UID, WorkKind, WorkPacket};
use struum_types::{Shardable, StruumError, split_ranges};

/// Tuning for [`crate::Node::run_sharded`].
#[derive(Debug, Clone)]
pub struct ShardOptions {
    /// Shards per eligible peer; more than 1 lets faster peers take more work.
    pub shards_per_peer: usize,
    /// Total tries per shard (first try + reassignments) before the job fails.
    pub max_attempts: u32,
    /// How long to wait for one shard's result before reassigning it.
    pub shard_timeout: Duration,
}

impl Default for ShardOptions {
    fn default() -> Self {
        Self {
            shards_per_peer: 2,
            max_attempts: 3,
            shard_timeout: Duration::from_secs(60),
        }
    }
}

/// What happened during a sharded run.
#[derive(Debug, Clone, Default)]
pub struct ShardReport {
    pub shards: usize,
    /// Shards completed by each peer (TCP address).
    pub completed_by: HashMap<SocketAddr, usize>,
    /// Peers dropped after a timeout or connection error.
    pub retired: Vec<SocketAddr>,
    /// Number of shard reassignments.
    pub retries: usize,
    pub elapsed: Duration,
}

/// Whether `peer` can run `kind` work: GPU work needs an OpenGL peer, any peer runs CPU work.
pub fn peer_accepts(peer: &NodeDetails, kind: WorkKind) -> bool {
    match kind {
        WorkKind::Gpu => peer.backend == Some(NodeBackend::OpenGL),
        WorkKind::Cpu => true,
    }
}

struct Shared {
    queue: Mutex<VecDeque<(usize, u32)>>,
    results: Mutex<Vec<Option<Work>>>,
    remaining: AtomicUsize,
    retries: AtomicUsize,
    /// Signalled when a shard completes or is requeued, waking idle peers.
    changed: Notify,
}

enum PeerExit {
    /// Queue drained, peer is healthy.
    Done,
    /// Peer failed and its shard was requeued.
    Retired(SocketAddr),
}

pub(crate) async fn run_sharded(
    sender_id: UID,
    work: Work,
    peers: Vec<PeerConnection>,
    opts: &ShardOptions,
) -> Result<(Work, ShardReport), StruumError> {
    let started = Instant::now();
    let kind = work.kind();
    let count = work.work_count();
    if count == 0 {
        return Err(StruumError::ShardError("work has no items (is the work buffer set?)".to_string()));
    }
    if peers.is_empty() {
        return Err(StruumError::ShardError(format!(
            "no connected peers can run {:?} work",
            kind
        )));
    }

    let ranges = split_ranges(count, peers.len() * opts.shards_per_peer.max(1));
    let expected: Vec<usize> = ranges.iter().map(|r| r.len()).collect();
    let payloads: Vec<Arc<Vec<u8>>> = ranges
        .iter()
        .map(|r| work.shard(r.clone()).and_then(|s| s.to_bytes()).map(Arc::new))
        .collect::<Result<_, _>>()?;
    drop(work);

    let shards = payloads.len();
    log::info!(
        "Sharding {:?} job ({} items) into {} shard(s) across {} peer(s)",
        kind,
        count,
        shards,
        peers.len()
    );

    let shared = Arc::new(Shared {
        queue: Mutex::new((0..shards).map(|i| (i, 0)).collect()),
        results: Mutex::new((0..shards).map(|_| None).collect()),
        remaining: AtomicUsize::new(shards),
        retries: AtomicUsize::new(0),
        changed: Notify::new(),
    });
    let payloads = Arc::new(payloads);
    let expected = Arc::new(expected);
    let job_id = uuid_like(sender_id);

    let mut set = JoinSet::new();
    for peer in peers {
        let shared = shared.clone();
        let payloads = payloads.clone();
        let expected = expected.clone();
        let opts = opts.clone();
        let job_id = job_id.clone();
        set.spawn(async move {
            peer_loop(peer, sender_id, kind, &job_id, &shared, &payloads, &expected, &opts).await
        });
    }

    let mut report = ShardReport { shards, ..Default::default() };
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok((exit, done))) => {
                if let PeerExit::Retired(addr) = exit {
                    report.retired.push(addr);
                }
                for addr in done {
                    *report.completed_by.entry(addr).or_default() += 1;
                }
            }
            Ok(Err(e)) => {
                set.abort_all();
                return Err(e);
            }
            Err(e) => {
                set.abort_all();
                return Err(StruumError::TaskError(format!("shard dispatcher panicked: {e}")));
            }
        }
    }

    let remaining = shared.remaining.load(Ordering::SeqCst);
    if remaining > 0 {
        return Err(StruumError::ShardError(format!(
            "{} of {} shard(s) unfinished: every eligible peer failed",
            remaining, shards
        )));
    }

    let parts: Vec<Work> = std::mem::take(&mut *shared.results.lock().await)
        .into_iter()
        .map(|r| r.expect("all shards completed"))
        .collect();
    let merged = Work::merge(parts)?;

    report.retries = shared.retries.load(Ordering::SeqCst);
    report.elapsed = started.elapsed();
    log::info!(
        "Sharded job finished in {:?}: {} shard(s), {} retr{}, per peer {:?}",
        report.elapsed,
        shards,
        report.retries,
        if report.retries == 1 { "y" } else { "ies" },
        report.completed_by
    );
    Ok((merged, report))
}

#[allow(clippy::too_many_arguments)]
async fn peer_loop(
    peer: PeerConnection,
    sender_id: UID,
    kind: WorkKind,
    job_id: &str,
    shared: &Shared,
    payloads: &[Arc<Vec<u8>>],
    expected: &[usize],
    opts: &ShardOptions,
) -> Result<(PeerExit, Vec<SocketAddr>), StruumError> {
    let mut done = Vec::new();
    loop {
        // Register for wakeups before checking state so no notification is missed.
        let changed = shared.changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();

        if shared.remaining.load(Ordering::SeqCst) == 0 {
            return Ok((PeerExit::Done, done));
        }
        let next = shared.queue.lock().await.pop_front();
        let Some((idx, attempt)) = next else {
            // Other peers hold the remaining shards; wait until one finishes or is requeued.
            changed.await;
            continue;
        };

        let packet = TcpPacket::WORK(WorkPacket {
            task_id: format!("{job_id}/{idx}"),
            sender_id,
            kind,
            data: payloads[idx].as_ref().clone(),
        });
        log::debug!("Shard {} (attempt {}) -> {}", idx, attempt + 1, peer.addr);

        let failure = match tokio::time::timeout(opts.shard_timeout, peer.request(&packet)).await {
            Ok(Ok(TcpPacket::RESULT(res))) => match Work::from_bytes(kind, &res.result_data) {
                Ok(work) if work.work_count() == expected[idx] => {
                    shared.results.lock().await[idx] = Some(work);
                    shared.remaining.fetch_sub(1, Ordering::SeqCst);
                    shared.changed.notify_waiters();
                    done.push(peer.addr);
                    continue;
                }
                Ok(work) => format!(
                    "returned {} items for shard {} (expected {})",
                    work.work_count(),
                    idx,
                    expected[idx]
                ),
                Err(e) => format!("sent an unreadable result: {e}"),
            },
            Ok(Ok(TcpPacket::WORKERROR(err))) => {
                return Err(StruumError::TaskError(format!(
                    "peer {} failed shard {}: {}",
                    peer.addr, idx, err.message
                )));
            }
            Ok(Ok(other)) => format!("sent an unexpected packet: {other:?}"),
            Ok(Err(e)) => format!("connection error: {e}"),
            Err(_) => format!("timed out after {:?}", opts.shard_timeout),
        };

        // Peer failure: requeue the shard for someone else and retire this peer.
        let attempts = attempt + 1;
        log::warn!(
            "Peer {} {} on shard {} (attempt {}/{}); reassigning",
            peer.addr,
            failure,
            idx,
            attempts,
            opts.max_attempts
        );
        if attempts >= opts.max_attempts {
            return Err(StruumError::ShardError(format!(
                "shard {} failed {} time(s); last error from {}: {}",
                idx, attempts, peer.addr, failure
            )));
        }
        shared.retries.fetch_add(1, Ordering::SeqCst);
        shared.queue.lock().await.push_back((idx, attempts));
        shared.changed.notify_waiters();
        return Ok((PeerExit::Retired(peer.addr), done));
    }
}

fn uuid_like(sender: UID) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("job-{}-{:x}", sender.inner, nanos)
}
