//! Sharded execution of work across connected peer nodes.
//!
//! The job's work range is split into `peers * shards_per_peer` shards, which
//! sit in a shared queue. One task per peer pulls shards, sends them as `WORK`
//! packets and collects the `RESULT`s, so faster peers take more shards.
//!
//! - Pipelining: each peer has up to `in_flight` shards outstanding. The worker
//!   runs them in order, so it starts the next shard while the previous result
//!   is still on its way back.
//! - Timeout / connection error: every shard the peer had in flight goes back on
//!   the queue for another peer (up to `max_attempts` tries) and that peer is retired.
//! - Speculation: once the queue is empty, an idle peer re-runs the oldest shard
//!   still out on another peer. The first result wins; the job returns without
//!   waiting for the slower copy.
//! - Joining peers: peers sent on the `joins` channel mid-job start pulling
//!   from the same queue (and can take speculative copies).
//! - `WORKERROR` (shader error, wasm trap, ...): deterministic, so the whole job fails.

use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, Semaphore, mpsc};
use tokio::task::JoinSet;

use crate::stats::NodeStats;

use struum_network::layer::{PeerConnection, PeerReader};
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
    /// How long to wait for each result, counted from when the shard was sent
    /// or the peer's previous result arrived, whichever is later.
    pub shard_timeout: Duration,
    /// Send large blobs (wasm modules, broadcast buffers) a peer already holds
    /// by hash instead of in full. See `TcpPacket::WORKREF`.
    pub blob_cache: bool,
    /// Shards sent to a peer ahead of its results. 2 lets the peer start the
    /// next shard while the previous result is on its way back; 1 is plain
    /// request/response.
    pub in_flight: usize,
    /// When the queue is empty, let an idle peer re-run the oldest shard still
    /// out on another peer, once that shard has taken longer than twice the
    /// mean round trip so far. At most one extra copy per shard.
    pub speculate: bool,
}

impl Default for ShardOptions {
    fn default() -> Self {
        Self {
            shards_per_peer: 2,
            max_attempts: 3,
            shard_timeout: Duration::from_secs(60),
            blob_cache: true,
            in_flight: 2,
            speculate: true,
        }
    }
}

/// What happened during a sharded run.
#[derive(Debug, Clone, Default)]
pub struct ShardReport {
    pub shards: usize,
    /// Shards completed by each peer (TCP address). Copies that lost a
    /// speculative race are not counted.
    pub completed_by: HashMap<SocketAddr, usize>,
    /// Peers dropped after a timeout or connection error.
    pub retired: Vec<SocketAddr>,
    /// Shards put back on the queue because their peer failed (every shard
    /// that peer had in flight).
    pub retries: usize,
    /// Work range of each shard, in order.
    pub ranges: Vec<Range<usize>>,
    /// Peer that completed each shard (same order as `ranges`).
    pub completed_on: Vec<SocketAddr>,
    /// Request-to-result time of the attempt that completed each shard (same
    /// order as `ranges`). With pipelining this includes time queued on the peer.
    pub round_trip: Vec<Duration>,
    /// Payload bytes sent to peers, including attempts that failed and
    /// speculative copies.
    pub bytes_sent: u64,
    /// Result payload bytes received from peers before the job finished.
    pub bytes_received: u64,
    /// Blob bytes not sent because the peer already held them.
    pub bytes_saved: u64,
    /// Shards resent in full because a peer had dropped a referenced blob.
    pub cache_misses: usize,
    /// Speculative copies sent to idle peers.
    pub speculative_runs: usize,
    /// Shards whose speculative copy finished before the original.
    pub speculative_wins: usize,
    pub elapsed: Duration,
}

/// Whether `peer` can run `kind` work: GPU work needs an OpenGL peer, any peer runs CPU work.
pub fn peer_accepts(peer: &NodeDetails, kind: WorkKind) -> bool {
    match kind {
        WorkKind::Gpu => peer.backend == Some(NodeBackend::OpenGL),
        WorkKind::Cpu => true,
    }
}

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Per-shard bookkeeping for the report, filled in as results arrive.
#[derive(Default)]
struct Completions {
    completed_on: Vec<Option<SocketAddr>>,
    round_trip: Vec<Duration>,
    completed_by: HashMap<SocketAddr, usize>,
    retired: Vec<SocketAddr>,
    rtt_sum: Duration,
    rtt_count: u32,
}

/// One job, shared by every peer task.
struct Job {
    sender_id: UID,
    kind: WorkKind,
    job_id: String,
    opts: ShardOptions,
    stats: Option<Arc<NodeStats>>,
    payloads: Vec<ShardPayload>,
    /// Item count of each shard's result.
    expected: Vec<usize>,
    /// Shards waiting for a peer, with the tries they have used.
    queue: std::sync::Mutex<VecDeque<(usize, u32)>>,
    results: std::sync::Mutex<Vec<Option<Work>>>,
    /// Shards out on some peer (not counting speculative copies), with when they were sent.
    running: std::sync::Mutex<HashMap<usize, Instant>>,
    /// Shards that already have a speculative copy.
    speculated: std::sync::Mutex<HashSet<usize>>,
    done: std::sync::Mutex<Completions>,
    remaining: AtomicUsize,
    retries: AtomicUsize,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    bytes_saved: AtomicU64,
    cache_misses: AtomicUsize,
    speculative_runs: AtomicUsize,
    speculative_wins: AtomicUsize,
    /// Signalled when a shard completes or is requeued, waking idle peers.
    changed: Notify,
}

/// What an idle peer can take when the queue is empty.
enum Spare {
    /// Re-run this shard now.
    Now(usize),
    /// The oldest running shard becomes worth re-running at this time.
    At(Instant),
    Nothing,
}

impl Job {
    /// Whether `idx` has a result. Also true once the job has returned and
    /// handed its results out, which peers still draining a slower copy can see.
    fn is_done(&self, idx: usize) -> bool {
        lock(&self.results).get(idx).is_none_or(Option::is_some)
    }

    /// Stores the first result for `idx`; later ones are dropped. Returns
    /// whether this one was first.
    fn complete(&self, idx: usize, work: Work, peer: SocketAddr, rtt: Duration, speculative: bool) -> bool {
        {
            let mut results = lock(&self.results);
            match results.get_mut(idx) {
                Some(slot @ None) => *slot = Some(work),
                _ => return false,
            }
        }
        lock(&self.running).remove(&idx);
        {
            let mut done = lock(&self.done);
            done.completed_on[idx] = Some(peer);
            done.round_trip[idx] = rtt;
            *done.completed_by.entry(peer).or_default() += 1;
            done.rtt_sum += rtt;
            done.rtt_count += 1;
        }
        if speculative {
            self.speculative_wins.fetch_add(1, Ordering::SeqCst);
        }
        self.remaining.fetch_sub(1, Ordering::SeqCst);
        self.changed.notify_waiters();
        true
    }

    /// Picks the oldest running shard without a speculative copy, once it has
    /// run longer than twice the mean round trip. Needs one completed shard to know
    /// what "longer" is.
    fn spare_shard(&self) -> Spare {
        let mean = {
            let done = lock(&self.done);
            if done.rtt_count == 0 {
                return Spare::Nothing;
            }
            // Twice the mean: peers of similar speed rarely trip it, real stragglers do.
            done.rtt_sum * 2 / done.rtt_count
        };
        let running = lock(&self.running);
        let mut speculated = lock(&self.speculated);
        let oldest = running
            .iter()
            .filter(|(idx, _)| !speculated.contains(*idx))
            .min_by_key(|(_, sent)| **sent)
            .map(|(idx, sent)| (*idx, *sent));
        match oldest {
            None => Spare::Nothing,
            Some((idx, sent)) if sent.elapsed() >= mean => {
                speculated.insert(idx);
                Spare::Now(idx)
            }
            Some((_, sent)) => Spare::At(sent + mean),
        }
    }
}

pub(crate) async fn run_sharded(
    sender_id: UID,
    work: Work,
    peers: Vec<PeerConnection>,
    mut joins: Option<mpsc::UnboundedReceiver<PeerConnection>>,
    opts: &ShardOptions,
    stats: Option<&Arc<NodeStats>>,
) -> Result<(Work, ShardReport), StruumError> {
    let started = Instant::now();
    let kind = work.kind();
    let count = work.work_count();
    if count == 0 {
        return Err(StruumError::ShardError("work has no items (is the work buffer set?)".to_string()));
    }
    if peers.is_empty() && joins.is_none() {
        return Err(StruumError::ShardError(format!(
            "no connected peers can run {:?} work",
            kind
        )));
    }

    // Shards are cut for the peers there at the start; ones joining later pull from the same queue.
    let ranges = split_ranges(count, peers.len().max(1) * opts.shards_per_peer.max(1));
    let expected: Vec<usize> = ranges.iter().map(|r| r.len()).collect();
    // Keep each shard's request: results leave out unchanged buffers, restored from it.
    let payloads: Vec<ShardPayload> = ranges
        .iter()
        .map(|r| {
            let work = work.shard(r.clone())?;
            let bytes = work.to_bytes()?;
            let blobs = if opts.blob_cache { work.blobs() } else { Vec::new() };
            let mut stripped = work.clone();
            for b in &blobs {
                stripped.take_blob(b.slot);
            }
            Ok(ShardPayload { work, bytes, blobs, stripped })
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

    let job = Arc::new(Job {
        sender_id,
        kind,
        job_id: uuid_like(sender_id),
        opts: opts.clone(),
        stats: stats.cloned(),
        payloads,
        expected,
        queue: std::sync::Mutex::new((0..shards).map(|i| (i, 0)).collect()),
        results: std::sync::Mutex::new((0..shards).map(|_| None).collect()),
        running: Default::default(),
        speculated: Default::default(),
        done: std::sync::Mutex::new(Completions {
            completed_on: vec![None; shards],
            round_trip: vec![Duration::ZERO; shards],
            ..Default::default()
        }),
        remaining: AtomicUsize::new(shards),
        retries: AtomicUsize::new(0),
        bytes_sent: AtomicU64::new(0),
        bytes_received: AtomicU64::new(0),
        bytes_saved: AtomicU64::new(0),
        cache_misses: AtomicUsize::new(0),
        speculative_runs: AtomicUsize::new(0),
        speculative_wins: AtomicUsize::new(0),
        changed: Notify::new(),
    });

    let mut set = JoinSet::new();
    for peer in peers {
        let job = job.clone();
        set.spawn(async move { peer_loop(peer, &job).await });
    }

    // Return as soon as every shard has a result, even if a slower copy is still out.
    loop {
        let changed = job.changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        if job.remaining.load(Ordering::SeqCst) == 0 {
            break;
        }
        tokio::select! {
            joined = set.join_next(), if !set.is_empty() || joins.is_none() => match joined {
                None => break,
                Some(Ok(Ok(()))) => {}
                Some(Ok(Err(e))) => {
                    set.abort_all();
                    return Err(e);
                }
                Some(Err(e)) => {
                    set.abort_all();
                    return Err(StruumError::TaskError(format!("shard dispatcher panicked: {e}")));
                }
            },
            _ = &mut changed => {}
            joined = async { joins.as_mut().expect("guarded").recv().await }, if joins.is_some() => match joined {
                Some(peer) => {
                    log::info!("Peer {} joined the job", peer.addr);
                    let job = job.clone();
                    set.spawn(async move { peer_loop(peer, &job).await });
                }
                None => joins = None,
            },
        }
    }
    // Peers still running a losing copy keep reading its reply in the
    // background, so their connection stays in step for the next job.
    set.detach_all();

    let remaining = job.remaining.load(Ordering::SeqCst);
    if remaining > 0 {
        return Err(StruumError::ShardError(format!(
            "{} of {} shard(s) unfinished: every eligible peer failed",
            remaining, shards
        )));
    }

    let parts: Vec<Work> = std::mem::take(&mut *lock(&job.results))
        .into_iter()
        .map(|r| r.expect("all shards completed"))
        .collect();
    let merged = Work::merge(parts)?;

    let done = std::mem::take(&mut *lock(&job.done));
    let report = ShardReport {
        shards,
        ranges,
        completed_by: done.completed_by,
        retired: done.retired,
        completed_on: done.completed_on.into_iter().map(|a| a.expect("all shards completed")).collect(),
        round_trip: done.round_trip,
        retries: job.retries.load(Ordering::SeqCst),
        bytes_sent: job.bytes_sent.load(Ordering::SeqCst),
        bytes_received: job.bytes_received.load(Ordering::SeqCst),
        bytes_saved: job.bytes_saved.load(Ordering::SeqCst),
        cache_misses: job.cache_misses.load(Ordering::SeqCst),
        speculative_runs: job.speculative_runs.load(Ordering::SeqCst),
        speculative_wins: job.speculative_wins.load(Ordering::SeqCst),
        elapsed: started.elapsed(),
    };
    log::info!(
        "Sharded job finished in {:?}: {} shard(s), {} retr{}, {} speculative win(s), per peer {:?}",
        report.elapsed,
        shards,
        report.retries,
        if report.retries == 1 { "y" } else { "ies" },
        report.speculative_wins,
        report.completed_by
    );
    Ok((merged, report))
}

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


/// A shard sent to a peer whose reply hasn't arrived yet.
#[derive(Clone)]
struct InFlight {
    idx: usize,
    /// Tries the shard had used before this one.
    attempt: u32,
    speculative: bool,
    task_id: String,
    sent_at: Instant,
    sent: u64,
    saved: u64,
}

/// State shared by one peer's send and receive loops.
#[derive(Default)]
struct PeerState {
    /// Sent shards in send order, which is the order the peer replies in.
    in_flight: std::sync::Mutex<VecDeque<InFlight>>,
    /// The peer failed: both loops stop.
    stop: AtomicBool,
    /// The send loop has finished; nothing more will be added to `in_flight`.
    sender_done: AtomicBool,
    /// Signalled whenever any of the above changes.
    wake: Notify,
}

impl PeerState {
    fn halt(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.wake.notify_waiters();
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// Resolves once [`PeerState::halt`] has been called.
    async fn halted(&self) {
        loop {
            let woke = self.wake.notified();
            tokio::pin!(woke);
            woke.as_mut().enable();
            if self.stopped() {
                return;
            }
            woke.await;
        }
    }
}

/// Runs one peer for one job: a send loop keeps up to `in_flight` shards
/// outstanding while a receive loop takes the replies in order.
async fn peer_loop(peer: PeerConnection, job: &Job) -> Result<(), StruumError> {
    // Held for the whole job, so replies to this job's shards reach only this
    // task. If an earlier job left a slower copy running here, its task still
    // holds the reader; we wait for it, which is also when the peer gets free.
    let mut reader = peer.reader().await;
    let state = PeerState::default();
    let window = Semaphore::new(job.opts.in_flight.max(1));

    let (sent, received) = tokio::join!(
        async {
            let out = send_loop(&peer, job, &state, &window).await;
            if !matches!(out, Ok(None)) {
                state.halt();
            }
            state.sender_done.store(true, Ordering::SeqCst);
            state.wake.notify_waiters();
            out
        },
        async {
            let out = recv_loop(&peer, &mut reader, job, &state, &window).await;
            if !matches!(out, Ok(None)) {
                state.halt();
            }
            out
        },
    );
    let failure = match (received?, sent?) {
        (Some(why), _) | (None, Some(why)) => why,
        (None, None) => return Ok(()),
    };

    // Peer failure: put everything it held back on the queue and retire it.
    let held: Vec<InFlight> = lock(&state.in_flight).drain(..).collect();
    if let (Some(head), Some(stats)) = (held.first(), &job.stats) {
        stats.used(peer.addr, false, head.sent, 0, head.sent_at.elapsed());
    }
    let mut requeue = Vec::new();
    for (i, shard) in held.into_iter().enumerate() {
        // Speculative copies aren't retried: the original is still out elsewhere.
        if shard.speculative || job.is_done(shard.idx) {
            continue;
        }
        lock(&job.running).remove(&shard.idx);
        // The shard that was due counts a failed try; ones queued behind it don't.
        let attempts = shard.attempt + u32::from(i == 0);
        log::warn!(
            "Peer {} {} on shard {} (attempt {}/{}); reassigning",
            peer.addr,
            failure,
            shard.idx,
            shard.attempt + 1,
            job.opts.max_attempts
        );
        if attempts >= job.opts.max_attempts {
            return Err(StruumError::ShardError(format!(
                "shard {} failed {} time(s); last error from {}: {}",
                shard.idx, attempts, peer.addr, failure
            )));
        }
        requeue.push((shard.idx, attempts));
    }
    job.retries.fetch_add(requeue.len(), Ordering::SeqCst);
    lock(&job.queue).extend(requeue);
    lock(&job.done).retired.push(peer.addr);
    job.changed.notify_waiters();
    Ok(())
}

/// Sends shards while the window has room. `Ok(Some(why))` means the peer failed.
async fn send_loop(
    peer: &PeerConnection,
    job: &Job,
    state: &PeerState,
    window: &Semaphore,
) -> Result<Option<String>, StruumError> {
    loop {
        let Some((idx, attempt, speculative)) = next_shard(job, state, window).await else {
            return Ok(None);
        };

        let payload = &job.payloads[idx];
        let task_id = format!("{}/{}", job.job_id, idx);
        let known = peer.known_blobs().clone();
        let (header, body, saved) = payload.encode_for(&known, task_id.clone(), job.sender_id, job.kind)?;
        log::debug!(
            "Shard {} ({}attempt {}) -> {} ({} bytes)",
            idx,
            if speculative { "speculative, " } else { "" },
            attempt + 1,
            peer.addr,
            body.len()
        );
        job.bytes_sent.fetch_add(body.len() as u64, Ordering::Relaxed);
        job.bytes_saved.fetch_add(saved, Ordering::Relaxed);
        // The peer keeps every blob it receives, inline or already held.
        peer.known_blobs().extend(payload.blobs.iter().map(|b| b.hash));

        let sent_at = Instant::now();
        if speculative {
            job.speculative_runs.fetch_add(1, Ordering::SeqCst);
        } else {
            lock(&job.running).insert(idx, sent_at);
        }
        lock(&state.in_flight).push_back(InFlight {
            idx,
            attempt,
            speculative,
            task_id,
            sent_at,
            sent: body.len() as u64,
            saved,
        });
        state.wake.notify_waiters();

        // A peer that stops reading can block this write forever, so give up on it when halted.
        tokio::select! {
            sent = peer.send_with_payload(&header, &body) => {
                if let Err(e) = sent {
                    return Ok(Some(format!("connection error: {e}")));
                }
            }
            _ = state.halted() => return Ok(None),
        }
    }
}

/// Waits for room in the window and a shard to send: the next queued one, or
/// a speculative copy when this peer is idle and the queue is empty. `None`
/// once the job is finished or the peer has failed.
async fn next_shard(job: &Job, state: &PeerState, window: &Semaphore) -> Option<(usize, u32, bool)> {
    loop {
        // Register for wakeups before checking state so no notification is missed.
        let changed = job.changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        if state.stopped() || job.remaining.load(Ordering::SeqCst) == 0 {
            return None;
        }

        let mut retry_at = None;
        let full = window.available_permits() == 0;
        if let Ok(permit) = window.try_acquire() {
            let queued = lock(&job.queue).pop_front();
            if let Some((idx, attempt)) = queued {
                permit.forget();
                return Some((idx, attempt, false));
            }
            if job.opts.speculate && lock(&state.in_flight).is_empty() {
                match job.spare_shard() {
                    Spare::Now(idx) => {
                        permit.forget();
                        return Some((idx, 0, true));
                    }
                    Spare::At(at) => retry_at = Some(at),
                    Spare::Nothing => {}
                }
            }
        }

        // Window full or nothing to send: wait for a reply, a requeue, or the spare time.
        let reply = window.acquire();
        tokio::select! {
            _ = &mut changed => {}
            _ = reply, if full => {}
            _ = state.halted() => return None,
            _ = tokio::time::sleep_until(retry_at.unwrap_or_else(Instant::now).into()), if retry_at.is_some() => {}
        }
    }
}

/// Takes the peer's replies in order. `Ok(Some(why))` means the peer failed;
/// the failed shards are left in `state.in_flight`.
async fn recv_loop(
    peer: &PeerConnection,
    reader: &mut PeerReader<'_>,
    job: &Job,
    state: &PeerState,
    window: &Semaphore,
) -> Result<Option<String>, StruumError> {
    let timeout = job.opts.shard_timeout;
    let mut last_reply = Instant::now();
    loop {
        let woke = state.wake.notified();
        tokio::pin!(woke);
        woke.as_mut().enable();
        if state.stopped() {
            return Ok(None);
        }
        let head = lock(&state.in_flight).front().cloned();
        let Some(head) = head else {
            if state.sender_done.load(Ordering::SeqCst) {
                return Ok(None);
            }
            woke.await;
            continue;
        };

        // The peer runs shards in order, so the clock for this one starts when
        // it was sent or when the previous result came back, whichever is later.
        let deadline = head.sent_at.max(last_reply) + timeout;
        let reply = tokio::select! {
            reply = tokio::time::timeout_at(deadline.into(), reader.recv()) => reply,
            _ = state.halted() => return Ok(None),
        };
        last_reply = Instant::now();

        let packet = match reply {
            Ok(Ok(packet)) => packet,
            Ok(Err(e)) => return Ok(Some(format!("connection error: {e}"))),
            Err(_) => return Ok(Some(format!("timed out after {timeout:?}"))),
        };
        let reply_id = match &packet {
            TcpPacket::RESULT(r) => Some(&r.job_id),
            TcpPacket::WORKERROR(e) => Some(&e.task_id),
            TcpPacket::CACHEMISS(m) => Some(&m.task_id),
            _ => None,
        };
        match reply_id {
            None => return Ok(Some(format!("sent an unexpected packet: {packet:?}"))),
            Some(id) if *id != head.task_id => {
                return Ok(Some(format!("replied to '{id}' while '{}' was due", head.task_id)));
            }
            Some(_) => {}
        }

        let received = match &packet {
            TcpPacket::RESULT(res) => res.result_data.len() as u64,
            _ => 0,
        };
        let record = |ok: bool| {
            if let Some(stats) = &job.stats {
                stats.used(peer.addr, ok, head.sent, received, head.sent_at.elapsed());
            }
        };
        let idx = head.idx;
        match packet {
            TcpPacket::CACHEMISS(miss) => {
                // The peer dropped some blobs: forget them and send this shard again in full.
                log::debug!("Peer {} missed {} blob(s) for shard {}", peer.addr, miss.missing.len(), idx);
                {
                    let mut known = peer.known_blobs();
                    for h in &miss.missing {
                        known.remove(h);
                    }
                }
                job.cache_misses.fetch_add(1, Ordering::SeqCst);
                job.bytes_saved.fetch_sub(head.saved, Ordering::Relaxed);
                if !head.speculative && !job.is_done(idx) {
                    lock(&job.running).remove(&idx);
                    lock(&job.queue).push_front((idx, head.attempt));
                }
                job.changed.notify_waiters();
            }
            TcpPacket::RESULT(res) => {
                let work = Work::from_bytes(job.kind, &res.result_data)
                    .and_then(|mut w| w.restore_elided(&job.payloads[idx].work).map(|()| w));
                let bad = match work {
                    Ok(work) if work.work_count() == job.expected[idx] => {
                        record(true);
                        if job.remaining.load(Ordering::SeqCst) > 0 {
                            job.bytes_received.fetch_add(received, Ordering::Relaxed);
                        }
                        let first = job.complete(idx, work, peer.addr, head.sent_at.elapsed(), head.speculative);
                        if !first {
                            log::debug!("Dropping later result for shard {} from {}", idx, peer.addr);
                        }
                        None
                    }
                    Ok(work) => Some(format!(
                        "returned {} items for shard {} (expected {})",
                        work.work_count(),
                        idx,
                        job.expected[idx]
                    )),
                    Err(e) => Some(format!("sent an unreadable result: {e}")),
                };
                if let Some(why) = bad {
                    record(false);
                    return Ok(Some(why));
                }
            }
            TcpPacket::WORKERROR(err) => {
                record(false);
                if !job.is_done(idx) {
                    return Err(StruumError::TaskError(format!(
                        "peer {} failed shard {}: {}",
                        peer.addr, idx, err.message
                    )));
                }
            }
            _ => unreachable!("reply_id matched only replies"),
        }

        lock(&state.in_flight).pop_front();
        window.add_permits(1);
        state.wake.notify_waiters();
    }
}

fn uuid_like(sender: UID) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("job-{}-{:x}", sender.inner, nanos)
}
