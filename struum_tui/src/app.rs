//! Screen state built from stats snapshots and the peer list; no terminal needed.

use std::collections::{HashMap, VecDeque};
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use struum_node::{PeerTally, StatsSnapshot, Tally};
use struum_types::network::{NodeBackend, UID};

use crate::discovery::{Discovery, RelayStatus};

/// Static facts about this node.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: UID,
    pub ip: IpAddr,
    pub udp_port: u16,
    pub tcp_port: u16,
    pub backend: NodeBackend,
}

/// One neighbour line: a peer the relay introduced and/or one we exchanged work with.
#[derive(Debug, Clone, PartialEq)]
pub struct PeerRow {
    /// `None` for traffic from an address the relay never introduced.
    pub id: Option<UID>,
    pub addr: String,
    pub backend: Option<NodeBackend>,
    pub last_seen: Option<Duration>,
    /// Shards we ran for this peer.
    pub for_them: Tally,
    /// Shards this peer ran for us.
    pub for_us: Tally,
}

/// How the neighbour table is ordered; Tab cycles through these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortKey {
    #[default]
    ForThem,
    ForUs,
    LastSeen,
    Id,
}

impl SortKey {
    pub fn next(self) -> Self {
        match self {
            SortKey::ForThem => SortKey::ForUs,
            SortKey::ForUs => SortKey::LastSeen,
            SortKey::LastSeen => SortKey::Id,
            SortKey::Id => SortKey::ForThem,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            SortKey::ForThem => "shards for them",
            SortKey::ForUs => "shards for us",
            SortKey::LastSeen => "last seen",
            SortKey::Id => "id",
        }
    }
}

/// Per-second samples of the served counters for the throughput sparkline.
#[derive(Debug, Default)]
pub struct History {
    last: Option<(Instant, u64, u64)>,
    /// Shards served per sample interval, newest last.
    pub shards: VecDeque<u64>,
    /// Bytes served (in + out) per sample interval, newest last.
    pub bytes: VecDeque<u64>,
}

impl History {
    pub const LEN: usize = 60;

    /// Adds a sample if at least `every` passed since the last one.
    pub fn sample(&mut self, now: Instant, served: &Tally, every: Duration) {
        let bytes = served.bytes_in + served.bytes_out;
        match self.last {
            Some((at, _, _)) if now.duration_since(at) < every => {}
            Some((_, shards0, bytes0)) => {
                push(&mut self.shards, served.shards.saturating_sub(shards0));
                push(&mut self.bytes, bytes.saturating_sub(bytes0));
                self.last = Some((now, served.shards, bytes));
            }
            None => self.last = Some((now, served.shards, bytes)),
        }
    }
}

fn push(q: &mut VecDeque<u64>, v: u64) {
    if q.len() == History::LEN {
        q.pop_front();
    }
    q.push_back(v);
}

/// Everything the screen shows.
#[derive(Debug, Clone)]
pub struct AppState {
    pub node: NodeInfo,
    pub relay: SocketAddr,
    pub relay_status: RelayStatus,
    pub stats: StatsSnapshot,
    pub peers: Vec<PeerRow>,
    pub shards_per_sec: Vec<u64>,
    pub bytes_per_sec: Vec<u64>,
    pub logs: Vec<String>,
    pub sort: SortKey,
    pub show_help: bool,
}

impl AppState {
    /// Fraction of uptime spent executing shards for others.
    pub fn utilisation(&self) -> f64 {
        let up = self.stats.uptime.as_secs_f64();
        if up <= 0.0 { 0.0 } else { (self.stats.served.busy.as_secs_f64() / up).min(1.0) }
    }

    /// Known peers by backend: (OpenGL, Vulkan, Cpu, unknown).
    pub fn peers_by_backend(&self) -> [usize; 4] {
        let mut n = [0; 4];
        for p in self.peers.iter().filter(|p| p.id.is_some()) {
            let i = match p.backend {
                Some(NodeBackend::OpenGL) => 0,
                Some(NodeBackend::Vulkan) => 1,
                Some(NodeBackend::Cpu) => 2,
                None => 3,
            };
            n[i] += 1;
        }
        n
    }
}

fn add(t: &mut Tally, o: &Tally) {
    t.shards += o.shards;
    t.failed += o.failed;
    t.bytes_in += o.bytes_in;
    t.bytes_out += o.bytes_out;
    t.busy += o.busy;
}

/// Joins introduced peers with the per-peer tallies.
///
/// `used_by_peer` is keyed by the worker's TCP address, which matches the
/// introduced peer exactly. `served_by_peer` is keyed by the requester's
/// ephemeral source address, so it matches by IP when exactly one known peer
/// has that IP; anything else gets its own row with no id.
pub fn neighbour_rows(discovery: &Discovery, stats: &StatsSnapshot, sort: SortKey) -> Vec<PeerRow> {
    let now = Instant::now();
    let mut rows: Vec<PeerRow> = Vec::new();
    let mut by_tcp: HashMap<SocketAddr, usize> = HashMap::new();
    let mut by_ip: HashMap<IpAddr, Vec<usize>> = HashMap::new();
    for p in discovery.peers.values() {
        let idx = rows.len();
        let tcp = p.details.tcp_addr();
        rows.push(PeerRow {
            id: Some(p.details.id),
            addr: tcp.map(|a| a.to_string()).unwrap_or_else(|| p.details.ip.to_string()),
            backend: p.details.backend,
            last_seen: Some(now.saturating_duration_since(p.last_seen)),
            for_them: Tally::default(),
            for_us: Tally::default(),
        });
        if let Some(tcp) = tcp {
            by_tcp.insert(tcp, idx);
        }
        by_ip.entry(p.details.ip.ip()).or_default().push(idx);
    }

    let mut unmatched: HashMap<SocketAddr, PeerRow> = HashMap::new();
    let mut place = |rows: &mut Vec<PeerRow>, pt: &PeerTally, exact_only: bool, for_us: bool| {
        let idx = by_tcp.get(&pt.peer).copied().or_else(|| {
            if exact_only {
                return None;
            }
            match by_ip.get(&pt.peer.ip()) {
                Some(v) if v.len() == 1 => Some(v[0]),
                _ => None,
            }
        });
        let row = match idx {
            Some(i) => &mut rows[i],
            None => unmatched.entry(pt.peer).or_insert_with(|| PeerRow {
                id: None,
                addr: pt.peer.to_string(),
                backend: None,
                last_seen: Some(pt.idle),
                for_them: Tally::default(),
                for_us: Tally::default(),
            }),
        };
        add(if for_us { &mut row.for_us } else { &mut row.for_them }, &pt.tally);
    };
    for pt in &stats.used_by_peer {
        place(&mut rows, pt, true, true);
    }
    for pt in &stats.served_by_peer {
        place(&mut rows, pt, false, false);
    }
    rows.extend(unmatched.into_values());

    rows.sort_by(|a, b| {
        let primary = match sort {
            SortKey::ForThem => b.for_them.shards.cmp(&a.for_them.shards),
            SortKey::ForUs => b.for_us.shards.cmp(&a.for_us.shards),
            SortKey::LastSeen => a.last_seen.cmp(&b.last_seen),
            SortKey::Id => a.id.map(|u| u.inner).cmp(&b.id.map(|u| u.inner)),
        };
        primary.then_with(|| a.addr.cmp(&b.addr))
    });
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use struum_types::network::{NodeDetails, NodeRole};

    fn details(id: u64, ip: [u8; 4], udp: u16, tcp: u16, backend: NodeBackend) -> NodeDetails {
        NodeDetails {
            id: UID::new(id),
            ip: SocketAddr::from((ip, udp)),
            role: NodeRole::NODE,
            backend: Some(backend),
            tcp_port: Some(tcp),
        }
    }

    fn tally(shards: u64) -> Tally {
        Tally { shards, bytes_in: shards * 10, bytes_out: shards, ..Tally::default() }
    }

    fn pt(addr: SocketAddr, shards: u64) -> PeerTally {
        PeerTally { peer: addr, tally: tally(shards), idle: Duration::ZERO }
    }

    #[test]
    fn rows_join_tallies_by_tcp_addr_and_unique_ip() {
        let mut d = Discovery::new("127.0.0.1:39001".parse().unwrap());
        d.introduce(UID::new(1), details(2, [10, 0, 0, 2], 5000, 6000, NodeBackend::Cpu));
        d.introduce(UID::new(1), details(3, [10, 0, 0, 3], 5000, 6000, NodeBackend::OpenGL));
        d.introduce(UID::new(1), details(1, [10, 0, 0, 1], 5000, 6000, NodeBackend::Cpu)); // ourselves
        let stats = StatsSnapshot {
            // Node 3 ran 4 shards for us (exact TCP address).
            used_by_peer: vec![pt("10.0.0.3:6000".parse().unwrap(), 4)],
            // Node 2 sent us work from an ephemeral port; 10.0.0.9 is unknown.
            served_by_peer: vec![
                pt("10.0.0.2:51234".parse().unwrap(), 7),
                pt("10.0.0.9:40000".parse().unwrap(), 1),
            ],
            ..StatsSnapshot::default()
        };

        let rows = neighbour_rows(&d, &stats, SortKey::ForThem);
        assert_eq!(rows.len(), 3, "{rows:#?}");
        assert_eq!(rows[0].id, Some(UID::new(2)));
        assert_eq!(rows[0].for_them.shards, 7);
        assert_eq!(rows[1].id, None);
        assert_eq!(rows[1].addr, "10.0.0.9:40000");
        assert_eq!(rows[2].id, Some(UID::new(3)));
        assert_eq!(rows[2].for_us.shards, 4);

        let by_us = neighbour_rows(&d, &stats, SortKey::ForUs);
        assert_eq!(by_us[0].id, Some(UID::new(3)));
    }

    #[test]
    fn ambiguous_ips_stay_unmatched() {
        // Two peers on one host (local testing): an ephemeral port can't be attributed.
        let mut d = Discovery::new("127.0.0.1:39001".parse().unwrap());
        d.introduce(UID::new(1), details(2, [127, 0, 0, 1], 5000, 6000, NodeBackend::Cpu));
        d.introduce(UID::new(1), details(3, [127, 0, 0, 1], 5001, 6001, NodeBackend::Cpu));
        let stats = StatsSnapshot {
            served_by_peer: vec![pt("127.0.0.1:50000".parse().unwrap(), 2)],
            ..StatsSnapshot::default()
        };
        let rows = neighbour_rows(&d, &stats, SortKey::ForThem);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].id, None);
        assert_eq!(rows[0].for_them.shards, 2);
    }

    #[test]
    fn history_samples_per_interval() {
        let mut h = History::default();
        let t0 = Instant::now();
        let s = Duration::from_secs(1);
        h.sample(t0, &tally(0), s);
        h.sample(t0 + Duration::from_millis(500), &tally(3), s); // too early
        h.sample(t0 + s, &tally(3), s);
        h.sample(t0 + 2 * s, &tally(10), s);
        assert_eq!(h.shards, [3, 7]);
        assert_eq!(h.bytes, [33, 77]);
        for i in 0..100 {
            h.sample(t0 + (3 + i) * s, &tally(10), s);
        }
        assert_eq!(h.shards.len(), History::LEN);
    }
}
