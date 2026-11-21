//! Sharded execution of work across connected peer nodes.
//!
//! The job's work range is split into `peers * shards_per_peer` shards, which
//! sit in a shared queue. One task per peer pulls shards, sends them as `WORK`
//! packets and waits for the `RESULT`, so faster peers take more shards.
//!
//! - Timeout / connection error: the shard goes back on the queue for another
//!   peer (up to `max_attempts` tries) and that peer is retired.
//! - `WORKERROR` (shader error, wasm trap, ...): deterministic, so the whole job fails.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, Notify};
use tokio::task::JoinSet;

use crate::stats::NodeStats;

use struum_network::layer::PeerConnection;
use struum_scheduler::Work;
use struum_types::network::{
    BlobRef, NodeBackend, NodeDetails, TcpPacket, UID, WorkKind, WorkPacket, WorkRefPacket,
};
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
    /// Send large blobs (wasm modules, broadcast buffers) a peer already holds
    /// by hash instead of in full. See `TcpPacket::WORKREF`.
    pub blob_cache: bool,
}

impl Default for ShardOptions {
    fn default() -> Self {
        Self {
            shards_per_peer: 2,
            max_attempts: 3,
            shard_timeout: Duration::from_secs(60),
            blob_cache: true,
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
    /// Blob bytes not sent because the peer already held them.
    pub bytes_saved: u64,
    /// Shards resent in full because a peer had dropped a referenced blob.
    pub cache_misses: usize,
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
    bytes_saved: AtomicU64,
    cache_misses: AtomicUsize,
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
    stats: Option<&Arc<NodeStats>>,
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
            let blobs = if opts.blob_cache { work.blobs() } else { Vec::new() };
            let mut stripped = work.clone();
            for b in &blobs {
                stripped.take_blob(b.slot);
            }
            Ok(Arc::new(ShardPayload { work, bytes, blobs, stripped }))
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
        bytes_saved: AtomicU64::new(0),
        cache_misses: AtomicUsize::new(0),
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
        let stats = stats.cloned();
        set.spawn(async move {
            peer_loop(peer, sender_id, kind, &job_id, &shared, &payloads, &expected, &opts, stats.as_deref())
                .await
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
    report.bytes_saved = shared.bytes_saved.load(Ordering::SeqCst);
    report.cache_misses = shared.cache_misses.load(Ordering::SeqCst);
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
    /// `work` serialized in full.
    bytes: Vec<u8>,
    /// Blobs of `work` worth sending by reference.
    blobs: Vec<BlobRef>,
    /// `work` with every blob in `blobs` emptied.
    stripped: Work,
}

impl ShardPayload {
    /// The packet header and payload to send to a peer holding `known` blobs:
    /// plain `WORK` if it holds none of them, else `WORKREF` with the held ones
    /// left out. Returns the bytes saved.
    fn encode_for(
        &self,
        known: &HashSet<u64>,
        task_id: String,
        sender_id: UID,
        kind: WorkKind,
    ) -> Result<(TcpPacket, std::borrow::Cow<'_, [u8]>, u64), StruumError> {
        let (refs, inline): (Vec<BlobRef>, Vec<BlobRef>) =
            self.blobs.iter().partition(|b| known.contains(&b.hash));
        if refs.is_empty() {
            let header = TcpPacket::WORK(WorkPacket { task_id, sender_id, kind, data: Vec::new() });
            return Ok((header, std::borrow::Cow::Borrowed(&self.bytes), 0));
        }
        let mut work = self.stripped.clone();
        for b in &inline {
            let bytes = self.work.blob_bytes(b.slot).expect("blob slot exists").to_vec();
            work.put_blob(b.slot, bytes);
        }
        let saved = refs
            .iter()
            .map(|b| self.work.blob_bytes(b.slot).map_or(0, |d| d.len() as u64))
            .sum();
        let header = TcpPacket::WORKREF(WorkRefPacket { task_id, sender_id, kind, refs, data: Vec::new() });
        Ok((header, std::borrow::Cow::Owned(work.to_bytes()?), saved))
    }
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
    stats: Option<&NodeStats>,
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

        // Full shards are written straight from `payloads`, not copied into the packet.
        let payload = &payloads[idx];
        let known = peer.known_blobs().clone();
        let (header, body, saved) = payload.encode_for(&known, format!("{job_id}/{idx}"), sender_id, kind)?;
        log::debug!("Shard {} (attempt {}) -> {} ({} bytes)", idx, attempt + 1, peer.addr, body.len());

        shared.bytes_sent.fetch_add(body.len() as u64, Ordering::Relaxed);
        shared.bytes_saved.fetch_add(saved, Ordering::Relaxed);
        // The peer keeps every blob it receives, inline or already held.
        peer.known_blobs().extend(payload.blobs.iter().map(|b| b.hash));
        let sent_at = Instant::now();
        let reply = tokio::time::timeout(opts.shard_timeout, peer.request_with_payload(&header, &body)).await;
        if let Ok(Ok(TcpPacket::CACHEMISS(miss))) = &reply {
            // The peer dropped some blobs: forget them and send this shard again in full.
            log::debug!("Peer {} missed {} blob(s) for shard {}", peer.addr, miss.missing.len(), idx);
            {
                let mut known = peer.known_blobs();
                for h in &miss.missing {
                    known.remove(h);
                }
            }
            shared.cache_misses.fetch_add(1, Ordering::SeqCst);
            shared.bytes_saved.fetch_sub(saved, Ordering::Relaxed);
            shared.queue.lock().await.push_front((idx, attempt));
            shared.changed.notify_waiters();
            continue;
        }
        let received = match &reply {
            Ok(Ok(TcpPacket::RESULT(res))) => res.result_data.len() as u64,
            _ => 0,
        };
        shared.bytes_received.fetch_add(received, Ordering::Relaxed);
        let sent = body.len() as u64;
        let record = |ok: bool| {
            if let Some(stats) = stats {
                stats.used(peer.addr, ok, sent, received, sent_at.elapsed());
            }
        };
        let failure = match reply {
            Ok(Ok(TcpPacket::RESULT(res))) => match Work::from_bytes(kind, &res.result_data)
                .and_then(|mut w| w.restore_elided(&payloads[idx].work).map(|()| w))
            {
                Ok(work) if work.work_count() == expected[idx] => {
                    shared.results.lock().await[idx] = Some(work);
                    shared.remaining.fetch_sub(1, Ordering::SeqCst);
                    shared.changed.notify_waiters();
                    record(true);
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
                record(false);
                return Err(StruumError::TaskError(format!(
                    "peer {} failed shard {}: {}",
                    peer.addr, idx, err.message
                )));
            }
            Ok(Ok(other)) => format!("sent an unexpected packet: {other:?}"),
            Ok(Err(e)) => format!("connection error: {e}"),
            Err(_) => format!("timed out after {:?}", opts.shard_timeout),
        };

        record(false);
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
