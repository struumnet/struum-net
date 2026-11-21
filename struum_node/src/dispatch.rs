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
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
    /// Work range of each shard, in order.
    pub ranges: Vec<Range<usize>>,
    /// Peer that completed each shard (same order as `ranges`).
    pub completed_on: Vec<SocketAddr>,
    /// Request-to-result time of the successful attempt of each shard (same order as `ranges`).
    pub round_trip: Vec<Duration>,
    /// Payload bytes sent to peers, including attempts that failed.
    pub bytes_sent: u64,
    /// Result payload bytes received from peers.
    pub bytes_received: u64,
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
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
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
    // Keep each shard's request: results leave out unchanged buffers, restored from it.
    let payloads: Vec<Arc<ShardPayload>> = ranges
        .iter()
        .map(|r| {
            let work = work.shard(r.clone())?;
            let bytes = work.to_bytes()?;
            Ok(Arc::new(ShardPayload { work, bytes }))
        })
        .collect::<Result<_, StruumError>>()?;
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
        bytes_sent: AtomicU64::new(0),
        bytes_received: AtomicU64::new(0),
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

    let mut report = ShardReport { shards, ranges, ..Default::default() };
    let mut completed_on: Vec<Option<SocketAddr>> = vec![None; shards];
    let mut round_trip = vec![Duration::ZERO; shards];
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok(Ok((exit, done))) => {
                if let PeerExit::Retired(addr) = exit {
                    report.retired.push(addr);
                }
                for (idx, addr, rtt) in done {
                    *report.completed_by.entry(addr).or_default() += 1;
                    completed_on[idx] = Some(addr);
                    round_trip[idx] = rtt;
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

    report.completed_on = completed_on.into_iter().map(|a| a.expect("all shards completed")).collect();
    report.round_trip = round_trip;
    report.retries = shared.retries.load(Ordering::SeqCst);
    report.bytes_sent = shared.bytes_sent.load(Ordering::SeqCst);
    report.bytes_received = shared.bytes_received.load(Ordering::SeqCst);
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
/// One shard as sent to peers, plus the request itself for restoring elided buffers.
struct ShardPayload {
    work: Work,
    bytes: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
async fn peer_loop(
    peer: PeerConnection,
    sender_id: UID,
    kind: WorkKind,
    job_id: &str,
    shared: &Shared,
    payloads: &[Arc<ShardPayload>],
    expected: &[usize],
    opts: &ShardOptions,
) -> Result<(PeerExit, Vec<(usize, SocketAddr, Duration)>), StruumError> {
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

        // The shard bytes are written straight from `payloads`, not copied into the packet.
        let header = TcpPacket::WORK(WorkPacket {
            task_id: format!("{job_id}/{idx}"),
            sender_id,
            kind,
            data: Vec::new(),
        });
        log::debug!("Shard {} (attempt {}) -> {}", idx, attempt + 1, peer.addr);

        shared.bytes_sent.fetch_add(payloads[idx].bytes.len() as u64, Ordering::Relaxed);
        let sent_at = Instant::now();
        let reply = tokio::time::timeout(opts.shard_timeout, peer.request_with_payload(&header, &payloads[idx].bytes)).await;
        if let Ok(Ok(TcpPacket::RESULT(res))) = &reply {
            shared.bytes_received.fetch_add(res.result_data.len() as u64, Ordering::Relaxed);
        }
        let failure = match reply {
            Ok(Ok(TcpPacket::RESULT(res))) => match Work::from_bytes(kind, &res.result_data)
                .and_then(|mut w| w.restore_elided(&payloads[idx].work).map(|()| w))
            {
                Ok(work) if work.work_count() == expected[idx] => {
                    shared.results.lock().await[idx] = Some(work);
                    shared.remaining.fetch_sub(1, Ordering::SeqCst);
                    shared.changed.notify_waiters();
                    done.push((idx, peer.addr, sent_at.elapsed()));
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
