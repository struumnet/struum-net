//! Live counters of what a node gave to and got from the network.
//!
//! A [`Node`](crate::Node) updates its [`NodeStats`] as it serves shards for other
//! nodes ([`NodeStats::served`]) and as other nodes run shards of its own jobs
//! ([`NodeStats::used`]). [`NodeStats::snapshot`] is cheap and can be polled from
//! another task (e.g. a TUI refreshing a few times a second) through the `Arc`
//! returned by [`Node::stats`](crate::Node::stats).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Totals for one direction of work with one peer (or all peers).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Shards that completed.
    pub shards: u64,
    /// Shards that failed (task error, timeout or lost connection).
    pub failed: u64,
    /// Payload bytes received.
    pub bytes_in: u64,
    /// Payload bytes sent.
    pub bytes_out: u64,
    /// Time spent: executing for [`NodeStats::served`], request-to-result for
    /// [`NodeStats::used`].
    pub busy: Duration,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.shards += other.shards;
        self.failed += other.failed;
        self.bytes_in += other.bytes_in;
        self.bytes_out += other.bytes_out;
        self.busy += other.busy;
    }
}

/// A [`Tally`] with one peer, plus when it was last updated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerTally {
    pub peer: SocketAddr,
    pub tally: Tally,
    /// Time since the last shard with this peer.
    pub idle: Duration,
}

/// A consistent copy of a node's counters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatsSnapshot {
    /// Time since the node started.
    pub uptime: Duration,
    /// Work this node did for others, in total.
    pub served: Tally,
    /// Work this node did for each requester, most shards first.
    pub served_by_peer: Vec<PeerTally>,
    /// Work others did for this node, in total.
    pub used: Tally,
    /// Work each worker did for this node, most shards first.
    pub used_by_peer: Vec<PeerTally>,
    /// Sharded jobs this node submitted.
    pub jobs: u64,
    /// Shard reassignments across those jobs.
    pub retries: u64,
    /// Whether this node is executing a shard for someone right now.
    pub working: bool,
}

#[derive(Debug, Default)]
struct Inner {
    served: HashMap<SocketAddr, (Tally, Option<Instant>)>,
    used: HashMap<SocketAddr, (Tally, Option<Instant>)>,
    jobs: u64,
    retries: u64,
    working: u32,
}

/// Thread-safe counters; see the [module docs](self).
#[derive(Debug)]
pub struct NodeStats {
    started: Instant,
    inner: Mutex<Inner>,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self { started: Instant::now(), inner: Mutex::new(Inner::default()) }
    }
}

fn bump(map: &mut HashMap<SocketAddr, (Tally, Option<Instant>)>, peer: SocketAddr, t: &Tally) {
    let entry = map.entry(peer).or_default();
    entry.0.add(t);
    entry.1 = Some(Instant::now());
}

fn sorted(map: &HashMap<SocketAddr, (Tally, Option<Instant>)>) -> (Tally, Vec<PeerTally>) {
    let mut total = Tally::default();
    let mut peers: Vec<PeerTally> = map
        .iter()
        .map(|(peer, (tally, last))| {
            total.add(tally);
            PeerTally { peer: *peer, tally: *tally, idle: last.map(|t| t.elapsed()).unwrap_or_default() }
        })
        .collect();
    peers.sort_by(|a, b| b.tally.shards.cmp(&a.tally.shards).then(a.peer.cmp(&b.peer)));
    (total, peers)
}

impl NodeStats {
    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Records that this node started executing a shard for a requester.
    pub fn begin_serving(&self) {
        self.lock().working += 1;
    }

    /// Records a shard this node executed for `requester`.
    pub fn served(&self, requester: SocketAddr, ok: bool, bytes_in: u64, bytes_out: u64, busy: Duration) {
        let t = Tally { shards: ok as u64, failed: !ok as u64, bytes_in, bytes_out, busy };
        let mut inner = self.lock();
        inner.working = inner.working.saturating_sub(1);
        bump(&mut inner.served, requester, &t);
    }

    /// Records a shard `worker` executed for this node.
    pub fn used(&self, worker: SocketAddr, ok: bool, bytes_out: u64, bytes_in: u64, round_trip: Duration) {
        let t = Tally { shards: ok as u64, failed: !ok as u64, bytes_in, bytes_out, busy: round_trip };
        bump(&mut self.lock().used, worker, &t);
    }

    /// Records a finished sharded job.
    pub fn job_done(&self, retries: u64) {
        let mut inner = self.lock();
        inner.jobs += 1;
        inner.retries += retries;
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        let inner = self.lock();
        let (served, served_by_peer) = sorted(&inner.served);
        let (used, used_by_peer) = sorted(&inner.used);
        StatsSnapshot {
            uptime: self.started.elapsed(),
            served,
            served_by_peer,
            used,
            used_by_peer,
            jobs: inner.jobs,
            retries: inner.retries,
            working: inner.working > 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    #[test]
    fn tallies_split_by_direction_and_peer() {
        let stats = NodeStats::default();
        stats.begin_serving();
        assert!(stats.snapshot().working);
        stats.served(addr(1), true, 100, 10, Duration::from_millis(5));
        stats.begin_serving();
        stats.served(addr(1), false, 50, 0, Duration::from_millis(1));
        stats.used(addr(2), true, 200, 20, Duration::from_millis(7));
        stats.used(addr(3), true, 200, 20, Duration::from_millis(7));
        stats.used(addr(3), true, 200, 20, Duration::from_millis(7));
        stats.job_done(1);

        let s = stats.snapshot();
        assert!(!s.working);
        assert_eq!(s.served, Tally { shards: 1, failed: 1, bytes_in: 150, bytes_out: 10, busy: Duration::from_millis(6) });
        assert_eq!(s.used.shards, 3);
        assert_eq!(s.used.bytes_out, 600);
        assert_eq!(s.used_by_peer[0].peer, addr(3), "most shards first");
        assert_eq!((s.jobs, s.retries), (1, 1));
    }
}
