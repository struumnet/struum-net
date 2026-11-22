use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::mpsc::{self, Receiver};

use struum_kernel::Kernel;
use struum_network::layer::NetworkLayer;
use struum_network::layer::PeerConnection;
use struum_network::quic::{QuicLayer, QuicPeer};
use struum_scheduler::{BlobStore, JobId, JobScheduler, Work};
use struum_types::StruumError;
use struum_types::network::{
    HelloPacket, IntroductionPacket, NetworkCommunicator, NetworkCoordinator, NodeBackend,
    NodeDetails, NodeRole, ResultPacket, TaskDataPacket, TaskPacket, TcpPacket, UID, UdpPacket,
    WorkErrorPacket, WorkPacket, WorkRefPacket, CacheMissPacket,
};

use crate::dispatch::{self, ShardOptions, ShardReport};
use crate::stats::NodeStats;

/// Coordinator component embedded in a `Node`.
/// Maintains the peer network map independently and provides node selection/lookup logic.
#[derive(Debug, Default, Clone)]
pub struct Coordinator {
    /// Registered peer nodes in this network map.
    pub nodes: HashMap<UID, NodeDetails>,
    /// Cursor for round-robin node selection across registered peers.
    next_node: usize,
}

impl Coordinator {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::default(),
            next_node: 0,
        }
    }

    /// Registers a peer node in this network map.
    pub fn register_node(&mut self, details: NodeDetails) {
        log::info!(
            "Registered peer node {} ({}, backend: {:?})",
            details.id,
            details.ip,
            details.backend
        );
        self.nodes.insert(details.id, details);
    }

    /// Removes a node from this network map.
    pub fn deregister_node(&mut self, id: &UID) {
        log::info!("Deregistered peer node {}", id);
        self.nodes.remove(id);
    }

    /// Returns a reference to a registered peer node by UID.
    pub fn get_node(&self, id: &UID) -> Option<&NodeDetails> {
        self.nodes.get(id)
    }

    /// Returns all registered peer nodes in this network map.
    pub fn get_nodes(&self) -> Vec<NodeDetails> {
        self.nodes.values().cloned().collect()
    }

    /// Number of registered peer nodes in this network map.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Round-robin pick among registered peer nodes to balance work across peers.
    pub fn select_node(&mut self) -> Option<NodeDetails> {
        if self.nodes.is_empty() {
            return None;
        }
        let ids: Vec<UID> = self.nodes.keys().copied().collect();
        let id = ids[self.next_node % ids.len()];
        self.next_node = self.next_node.wrapping_add(1);
        self.nodes.get(&id).cloned()
    }

    /// Selects a registered peer node matching the requested computation backend.
    pub fn select_node_by_backend(&mut self, backend: NodeBackend) -> Option<NodeDetails> {
        self.nodes
            .values()
            .find(|node| node.backend == Some(backend))
            .cloned()
    }
}

/// Represents a computational Node in the network with an embedded Coordinator.
/// The embedded `coordinator` allows each node to store and manage its peer map independently.
pub struct Node<const BUF_SIZE: usize> {
    /// Id uniquely identifies the node.
    pub id: UID,
    /// Network layer with BUF_SIZE buffer for network communication.
    pub net: NetworkLayer<BUF_SIZE>,
    /// Defines the computation backend (e.g. CPU, OpenGL).
    pub backend: NodeBackend,
    /// Intra-node scheduler communicating and scheduling tasks across local worker threads.
    pub scheduler: JobScheduler,
    /// Channel receiver for completed job IDs from local worker threads.
    pub job_rx: Receiver<JobId>,
    /// Embedded Coordinator component managing known peer nodes.
    pub coordinator: Coordinator,
    /// Configured relay server address for registration polling.
    pub relay_addr: Option<SocketAddr>,
    /// Work given to and received from the network; see [`Node::stats`].
    stats: Arc<NodeStats>,
    /// Bytes of blobs kept per requester connection when serving (see
    /// `TcpPacket::WORKREF`). Defaults to `STRUUM_BLOB_CACHE_MB` (1024 MiB).
    pub blob_cache_bytes: usize,
    /// QUIC endpoint, once enabled ([`Node::enable_quic`]) or first needed to dial.
    quic: Option<Arc<QuicLayer>>,
    /// Authenticated QUIC connections to peers, by the peer's advertised QUIC address.
    quic_peers: HashMap<SocketAddr, QuicPeer>,
    /// Whether [`Node::details`] (what the relay tells requesters) includes the
    /// TCP port. Off when TCP isn't served, e.g. because a token is set.
    advertise_tcp: bool,
}

impl<const BUF_SIZE: usize> std::fmt::Debug for Node<BUF_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("net", &self.net)
            .field("backend", &self.backend)
            .field("peers_count", &self.coordinator.node_count())
            .field("relay_addr", &self.relay_addr)
            .finish()
    }
}

impl<const BUF_SIZE: usize> Node<BUF_SIZE> {
    /// Creates a new Node with an automatically generated UID.
    pub async fn new(
        udp_port: u16,
        tcp_port: u16,
        backend: NodeBackend,
        workers: u32,
    ) -> Result<Self, StruumError> {
        Self::with_id(UID::generate(), udp_port, tcp_port, backend, workers).await
    }

    /// Creates a new Node with a specified UID.
    pub async fn with_id(
        id: UID,
        udp_port: u16,
        tcp_port: u16,
        backend: NodeBackend,
        workers: u32,
    ) -> Result<Self, StruumError> {
        let (tx, job_rx) = mpsc::channel(BUF_SIZE);
        let scheduler = JobScheduler::new(tx, workers);
        let net = NetworkLayer::new(udp_port, tcp_port).await?;

        log::info!(
            "Initialized Node {} [IP: {}] [UDP: {}] [TCP: {}] [Backend: {:?}]",
            id,
            net.ip,
            udp_port,
            tcp_port,
            backend
        );

        Ok(Self {
            id,
            net,
            backend,
            scheduler,
            job_rx,
            coordinator: Coordinator::new(),
            relay_addr: None,
            stats: Arc::new(NodeStats::default()),
            blob_cache_bytes: BlobStore::env_limit_bytes(),
            quic: None,
            quic_peers: HashMap::new(),
            advertise_tcp: true,
        })
    }

    /// Live counters of shards this node ran for others and others ran for it.
    /// The handle stays valid while the node is moved into a serving task.
    pub fn stats(&self) -> Arc<NodeStats> {
        self.stats.clone()
    }

    // --- Intra-node thread communication methods (via struum_scheduler) ---

    /// Submits a kernel to be scheduled and executed by worker threads on this node.
    pub async fn submit_job(&self, kernel: Kernel) -> JobId {
        let id = self.scheduler.add_job(kernel).await;
        log::debug!("Node {} submitted job {} to local scheduler", self.id, id);
        id
    }

    /// Waits for the next job completed by one of the local worker threads.
    pub async fn recv_completed_job(&mut self) -> Option<JobId> {
        let id = self.job_rx.recv().await;
        if let Some(ref job_id) = id {
            log::debug!("Node {} received completion notice for job {}", self.id, job_id);
        }
        id
    }

    /// Retrieves the resulting Kernel of a completed job from the local scheduler.
    pub async fn get_job_result(&self, id: &JobId) -> Option<Kernel> {
        self.scheduler.get_result_of_job(id).await
    }

    /// Submits any kind of work (GPU kernel or CPU task) to the local scheduler.
    pub async fn submit_work(&self, work: impl Into<Work>) -> JobId {
        self.scheduler.add_work(work.into()).await
    }

    /// Result of a completed job: `Some(Ok(work))`, `Some(Err(message))`, or `None` if unknown.
    pub async fn get_work_result(&self, id: &JobId) -> Option<Result<Work, String>> {
        self.scheduler.get_work_result(id).await
    }

    /// Runs `work` on this node and waits for it to finish.
    pub async fn execute_local(&mut self, work: impl Into<Work>) -> Result<Work, StruumError> {
        self.scheduler.run(work.into()).await
    }

    // --- Sharded execution across peers (see `dispatch`) ---

    /// Starts a QUIC endpoint on UDP `port` (0 picks one) that peers must
    /// authenticate to with `token` (see `struum_network::quic`). Returns the
    /// port. Accepting is up to the caller: see [`Node::quic`].
    pub fn enable_quic(&mut self, port: u16, token: &[u8]) -> Result<u16, StruumError> {
        let layer = QuicLayer::bind(SocketAddr::from(([0, 0, 0, 0], port)), self.id.inner, token)?;
        let port = layer.local_addr()?.port();
        log::info!("Node {} serving QUIC on UDP {}", self.id, port);
        self.quic = Some(Arc::new(layer));
        Ok(port)
    }

    /// Stops (or resumes) advertising the TCP port to the relay. Turn it off
    /// when this node won't accept TCP, so requesters don't try it.
    pub fn set_advertise_tcp(&mut self, on: bool) {
        self.advertise_tcp = on;
    }

    /// The QUIC endpoint, if enabled: accept requesters with `quic().accept()`
    /// and serve each with [`PeerServer::serve`].
    pub fn quic(&self) -> Option<Arc<QuicLayer>> {
        self.quic.clone()
    }

    /// UDP port of the QUIC endpoint, if enabled.
    pub fn quic_port(&self) -> Option<u16> {
        self.quic.as_ref().and_then(|q| q.local_addr().ok()).map(|a| a.port())
    }

    /// Connects to every given peer: over QUIC if it advertises a QUIC port,
    /// otherwise over TCP. Returns the addresses connected (already-open ones
    /// included).
    ///
    /// A QUIC peer is never retried over TCP: if it rejects our token, it
    /// isn't one of ours. The first QUIC peer starts this node's QUIC endpoint
    /// (token from `STRUUM_TOKEN`) unless [`Node::enable_quic`] already did.
    pub async fn connect_to_peers(&mut self, peers: &[NodeDetails]) -> Vec<SocketAddr> {
        let mut connected = Vec::new();
        for peer in peers {
            if let Some(addr) = peer.quic_addr() {
                match self.connect_quic(addr).await {
                    Ok(()) => connected.push(addr),
                    Err(e) => log::warn!("Could not connect to peer {} over QUIC at {}: {}", peer.id, addr, e),
                }
                continue;
            }
            let Some(addr) = peer.tcp_addr() else {
                log::warn!("Peer {} has no TCP or QUIC port; skipping", peer.id);
                continue;
            };
            if self.net.peer(&addr).is_some() {
                connected.push(addr);
                continue;
            }
            match self.connect_to_peer_addr(addr).await {
                Ok(()) => connected.push(addr),
                Err(e) => log::warn!("Could not connect to peer {} at {}: {}", peer.id, addr, e),
            }
        }
        connected
    }

    async fn connect_quic(&mut self, addr: SocketAddr) -> Result<(), StruumError> {
        if self.quic_peers.get(&addr).is_some_and(|p| !p.conn.is_broken()) {
            return Ok(());
        }
        if self.quic.is_none() {
            self.enable_quic(0, &struum_network::quic::token_from_env())?;
        }
        let quic = self.quic.clone().expect("enabled above");
        let peer = quic.connect(addr).await?;
        log::info!("Node {} connected to peer {} over QUIC at {}", self.id, peer.id, addr);
        self.quic_peers.insert(addr, peer);
        Ok(())
    }

    /// Connected peers that can run `kind` work.
    fn eligible_peers(&self, kind: struum_types::network::WorkKind) -> Vec<PeerConnection> {
        self.coordinator
            .get_nodes()
            .into_iter()
            .filter(|p| dispatch::peer_accepts(p, kind))
            .filter_map(|p| match p.quic_addr() {
                Some(addr) => self.quic_peers.get(&addr).map(|q| q.conn.clone()),
                None => p.tcp_addr().and_then(|addr| self.net.peer(&addr)),
            })
            .filter(|peer| !peer.is_broken())
            .collect()
    }

    /// Splits `work` into shards, runs them on all connected eligible peers
    /// (GPU work on OpenGL peers, CPU work on any peer), reassigns shards from
    /// failed peers and merges the results in order.
    pub async fn run_sharded(
        &mut self,
        work: impl Into<Work>,
        opts: &ShardOptions,
    ) -> Result<Work, StruumError> {
        self.run_sharded_with_report(work, opts).await.map(|(w, _)| w)
    }

    /// Like [`Node::run_sharded`], also returning per-peer statistics.
    pub async fn run_sharded_with_report(
        &mut self,
        work: impl Into<Work>,
        opts: &ShardOptions,
    ) -> Result<(Work, ShardReport), StruumError> {
        self.run_sharded_inner(work.into(), None, opts).await
    }

    /// Like [`Node::run_sharded_with_report`], also taking peers that connect
    /// while the job runs: each one sent on `joins` starts pulling shards at
    /// once. The caller sends only peers that can run this kind of work. If
    /// every peer fails, the job waits for a joiner until the sender is dropped.
    pub async fn run_sharded_with_joins(
        &mut self,
        work: impl Into<Work>,
        joins: tokio::sync::mpsc::UnboundedReceiver<PeerConnection>,
        opts: &ShardOptions,
    ) -> Result<(Work, ShardReport), StruumError> {
        self.run_sharded_inner(work.into(), Some(joins), opts).await
    }

    async fn run_sharded_inner(
        &mut self,
        work: Work,
        joins: Option<tokio::sync::mpsc::UnboundedReceiver<PeerConnection>>,
        opts: &ShardOptions,
    ) -> Result<(Work, ShardReport), StruumError> {
        let peers = self.eligible_peers(work.kind());
        let result = dispatch::run_sharded(self.id, work, peers, joins, opts, Some(&self.stats)).await;
        if let Ok((_, report)) = &result {
            self.stats.job_done(report.retries as u64);
        }

        if let Ok((_, report)) = &result {
            // Forget peers that failed so later jobs skip them.
            for addr in &report.retired {
                self.net.disconnect(addr);
                if let Some(peer) = self.quic_peers.remove(addr) {
                    peer.close();
                }
                let ids: Vec<UID> = self
                    .coordinator
                    .get_nodes()
                    .into_iter()
                    .filter(|p| p.tcp_addr() == Some(*addr) || p.quic_addr() == Some(*addr))
                    .map(|p| p.id)
                    .collect();
                for id in ids {
                    self.coordinator.deregister_node(&id);
                }
            }
        }
        result
    }

    /// Serves one connected requester until it disconnects; see [`PeerServer::serve`].
    pub async fn serve_peer(&mut self, peer_addr: SocketAddr) -> Result<(), StruumError> {
        let peer = self
            .net
            .peer(&peer_addr)
            .ok_or_else(|| StruumError::NetworkConnectionError(format!("No TCP connection found for {peer_addr}")))?;
        self.server().serve(peer).await;
        self.net.disconnect(&peer_addr);
        Ok(())
    }

    /// A cloneable handle that serves requesters without borrowing the node,
    /// so each connection can be served from its own task:
    ///
    /// ```no_run
    /// # async fn run(mut node: struum_node::Node<2048>) -> Result<(), struum_types::StruumError> {
    /// loop {
    ///     let peer = node.accept_peer().await?;
    ///     tokio::spawn(node.server().serve(peer));
    /// }
    /// # }
    /// ```
    pub fn server(&self) -> PeerServer {
        PeerServer {
            id: self.id,
            scheduler: self.scheduler.clone(),
            stats: self.stats.clone(),
            blob_cache_bytes: self.blob_cache_bytes,
        }
    }

    /// Accepts the next requester connection, for [`PeerServer::serve`].
    pub async fn accept_peer(&self) -> Result<PeerConnection, StruumError> {
        self.net.accept_peer().await
    }

    // --- Inter-node peer communication methods ---

    /// Connects directly to a peer node over TCP.
    pub async fn connect_to_peer(&mut self, peer_ip: &str, peer_port: u16) -> Result<(), StruumError> {
        self.net.initialize_tcp_connection(peer_ip, peer_port).await
    }

    /// Connects directly to a peer node at the specified SocketAddr over TCP.
    pub async fn connect_to_peer_addr(&mut self, peer_addr: SocketAddr) -> Result<(), StruumError> {
        log::info!("Node {} connecting to peer at {} over TCP...", self.id, peer_addr);
        self.net.initialize_tcp_connection_addr(peer_addr).await?;
        log::info!("Node {} established TCP connection with peer at {}", self.id, peer_addr);
        Ok(())
    }

    /// Accepts an incoming TCP connection from a peer node.
    /// Returns the SocketAddr of the connected peer.
    pub async fn accept_peer_connection(&mut self) -> Result<SocketAddr, StruumError> {
        log::info!("Node {} waiting for incoming TCP peer connection...", self.id);
        let peer_addr = self.net.accept_tcp_connection().await?;
        log::info!("Node {} accepted TCP connection from peer at {}", self.id, peer_addr);
        Ok(peer_addr)
    }

    /// Sends a computational task (serialized Kernel with source and buffer bindings) to a peer over TCP.
    pub async fn send_task_data(
        &mut self,
        peer_addr: SocketAddr,
        task_id: &str,
        kernel: &Kernel,
    ) -> Result<(), StruumError> {
        log::debug!("Node {} sending task '{}' to {}", self.id, task_id, peer_addr);
        let kernel_data = bincode::serialize(kernel)
            .map_err(|e| StruumError::SerializationError(format!("Failed to serialize Kernel: {}", e)))?;
        let packet = TaskDataPacket::new(task_id.to_string(), self.id, kernel_data);
        self.net.send_tcp_data(peer_addr, TcpPacket::TASKDATA(packet)).await
    }

    /// Sends a Kernel directly over TCP to a peer for remote execution.
    pub async fn send_kernel(
        &mut self,
        peer_addr: SocketAddr,
        task_id: &str,
        kernel: &Kernel,
    ) -> Result<(), StruumError> {
        self.send_task_data(peer_addr, task_id, kernel).await
    }

    /// Receives a computational task (serialized Kernel with source and buffer bindings) from a peer over TCP.
    /// Returns (task_id, sender_id, kernel).
    pub async fn recv_task_data(
        &mut self,
        peer_addr: &SocketAddr,
    ) -> Result<(String, UID, Kernel), StruumError> {
        let packet = self.net.recv_tcp_data(peer_addr).await?;
        match packet {
            TcpPacket::TASKDATA(data_packet) => {
                log::debug!(
                    "Node {} received task '{}' from sender {}",
                    self.id,
                    data_packet.task_id,
                    data_packet.sender_id
                );
                let kernel: Kernel = bincode::deserialize(&data_packet.kernel_data)
                    .map_err(|e| StruumError::SerializationError(format!("Failed to deserialize Kernel: {}", e)))?;
                Ok((data_packet.task_id, data_packet.sender_id, kernel))
            }
            other => Err(StruumError::NetworkConnectionError(format!(
                "Expected TASKDATA packet, received: {:?}",
                other
            ))),
        }
    }

    /// Receives a Kernel sent over TCP by a peer.
    /// Returns (task_id, sender_id, kernel).
    pub async fn recv_kernel(
        &mut self,
        peer_addr: &SocketAddr,
    ) -> Result<(String, UID, Kernel), StruumError> {
        self.recv_task_data(peer_addr).await
    }

    /// Sends a computation result (serialized Kernel with output data) back to a peer over TCP.
    pub async fn send_task_result(
        &mut self,
        peer_addr: SocketAddr,
        job_id: &str,
        result_kernel: &Kernel,
    ) -> Result<(), StruumError> {
        log::debug!("Node {} sending task result for '{}' to {}", self.id, job_id, peer_addr);
        let result_data = bincode::serialize(result_kernel)
            .map_err(|e| StruumError::SerializationError(format!("Failed to serialize result Kernel: {}", e)))?;
        let packet = ResultPacket::new(job_id.to_string(), self.id, result_data);
        self.net.send_tcp_data(peer_addr, TcpPacket::RESULT(packet)).await
    }

    /// Receives a computation result from a peer over TCP.
    /// Returns (job_id, sender_id, kernel).
    pub async fn recv_task_result(
        &mut self,
        peer_addr: &SocketAddr,
    ) -> Result<(String, UID, Kernel), StruumError> {
        let packet = self.net.recv_tcp_data(peer_addr).await?;
        match packet {
            TcpPacket::RESULT(res) => {
                log::debug!(
                    "Node {} received task result for '{}' from sender {}",
                    self.id,
                    res.job_id,
                    res.sender_id
                );
                let kernel: Kernel = bincode::deserialize(&res.result_data)
                    .map_err(|e| StruumError::SerializationError(format!("Failed to deserialize result Kernel: {}", e)))?;
                Ok((res.job_id, res.sender_id, kernel))
            }
            other => Err(StruumError::NetworkConnectionError(format!(
                "Expected RESULT packet, received: {:?}",
                other
            ))),
        }
    }

    /// Sends any generic TcpPacket to a peer over TCP.
    pub async fn send_tcp_packet(
        &mut self,
        peer_addr: SocketAddr,
        packet: TcpPacket,
    ) -> Result<(), StruumError> {
        self.net.send_tcp_data(peer_addr, packet).await
    }

    /// Receives any generic TcpPacket from a peer over TCP.
    pub async fn recv_tcp_packet(
        &mut self,
        peer_addr: &SocketAddr,
    ) -> Result<TcpPacket, StruumError> {
        self.net.recv_tcp_data(peer_addr).await
    }

    /// Sends a computation task packet to a peer node over TCP.
    pub async fn send_task_to_peer(
        &mut self,
        peer_addr: SocketAddr,
        task: TaskPacket,
    ) -> Result<(), StruumError> {
        self.net.send_tcp_data(peer_addr, TcpPacket::TASK(task)).await
    }

    pub async fn long_poll_relay(&mut self, relay_addr: SocketAddr) -> Result<NodeDetails, StruumError> {
        self.long_poll_relay_with_interval(relay_addr, std::time::Duration::from_millis(1000)).await
    }

    /// Long-polls connection requests to the relay server with a configurable poll interval.
    /// If any peer already exists in the node's coordinator, relay polling is skipped.
    pub async fn long_poll_relay_with_interval(
        &mut self,
        relay_addr: SocketAddr,
        interval: std::time::Duration,
    ) -> Result<NodeDetails, StruumError> {
        // If an IP already exists in the coordinator, do not request the relay
        if let Some(existing_peer) = self.coordinator.select_node() {
            log::info!(
                "Node {} already has known peer {} in coordinator (IP: {}). Skipping relay request.",
                self.id,
                existing_peer.id,
                existing_peer.ip
            );
            return Ok(existing_peer);
        }

        self.relay_addr = Some(relay_addr);
        log::info!("Node {} long-polling connection request to relay at {}...", self.id, relay_addr);

        loop {
            if let Some(existing_peer) = self.coordinator.select_node() {
                return Ok(existing_peer);
            }

            let _ = self.introduce(relay_addr).await;

            tokio::select! {
                res = self.listen_introduction() => {
                    let intro = res?;
                    let details = NodeDetails::from(intro);
                    self.coordinator.register_node(details.clone());
                    log::info!(
                        "Node {} successfully connected to relay! Received peer {} at {} (TCP: {:?})",
                        self.id,
                        details.id,
                        details.ip,
                        details.tcp_port
                    );
                    return Ok(details);
                }
                _ = tokio::time::sleep(interval) => {
                    log::debug!("Node {} polling relay at {} for peer introduction...", self.id, relay_addr);
                }
            }
        }
    }

    pub async fn long_poll_relay_for_peers(
        &mut self,
        relay_addr: SocketAddr,
        min_peers: usize,
    ) -> Result<Vec<NodeDetails>, StruumError> {
        self.long_poll_relay_for_peers_with_interval(
            relay_addr,
            min_peers,
            std::time::Duration::from_millis(1000),
        ).await
    }

    /// Long-polls connection requests to the relay server until at least `min_peers` peer nodes
    /// are discovered with a configurable poll interval.
    /// If node already has at least `min_peers` in coordinator, relay polling is skipped.
    pub async fn long_poll_relay_for_peers_with_interval(
        &mut self,
        relay_addr: SocketAddr,
        min_peers: usize,
        interval: std::time::Duration,
    ) -> Result<Vec<NodeDetails>, StruumError> {
        if self.coordinator.node_count() >= min_peers {
            log::info!(
                "Node {} already has {} peer(s) in coordinator (target: {}). Skipping relay request.",
                self.id,
                self.coordinator.node_count(),
                min_peers
            );
            return Ok(self.coordinator.get_nodes());
        }

        self.relay_addr = Some(relay_addr);
        log::info!(
            "Node {} long-polling relay at {} for {} peer(s)...",
            self.id,
            relay_addr,
            min_peers
        );

        // Re-register with the relay only once per `interval`: each registration makes the relay
        // send introductions for every known node, so re-registering on every received packet
        // would turn into a tight request/introduction loop.
        let mut next_poll = tokio::time::Instant::now();
        while self.coordinator.node_count() < min_peers {
            if tokio::time::Instant::now() >= next_poll {
                let _ = self.introduce(relay_addr).await;
                next_poll = tokio::time::Instant::now() + interval;
            }

            tokio::select! {
                res = self.listen_introduction() => {
                    if let Ok(intro) = res {
                        let details = NodeDetails::from(intro);
                        self.coordinator.register_node(details);
                    }
                }
                _ = tokio::time::sleep_until(next_poll) => {
                    log::debug!(
                        "Node {} polling relay at {} ({}/{} peers discovered)...",
                        self.id,
                        relay_addr,
                        self.coordinator.node_count(),
                        min_peers
                    );
                }
            }
        }

        Ok(self.coordinator.get_nodes())
    }

    /// Waits via the relay until at least `min_peers` peers are known, then keeps
    /// collecting introductions for `settle` so peers that arrive slightly later
    /// are included too. Returns every known peer.
    pub async fn discover_peers(
        &mut self,
        relay_addr: SocketAddr,
        min_peers: usize,
        settle: std::time::Duration,
    ) -> Result<Vec<NodeDetails>, StruumError> {
        self.long_poll_relay_for_peers(relay_addr, min_peers).await?;
        // Polling for an unreachable count just listens until `settle` elapses;
        // introductions received meanwhile are registered in the coordinator.
        let _ = tokio::time::timeout(settle, self.long_poll_relay_for_peers(relay_addr, usize::MAX)).await;
        Ok(self.coordinator.get_nodes())
    }

    /// Listens for an introduction packet sent by the coordinator or relay introducing a sibling node.
    /// If any peer already exists in the coordinator, returns it immediately without contacting the relay.
    pub async fn listen_sibling_introduction(&mut self) -> Result<NodeDetails, StruumError> {
        if let Some(existing_peer) = self.coordinator.select_node() {
            log::info!(
                "Node {} already has known peer {} in coordinator (IP: {}). Skipping relay introduction.",
                self.id,
                existing_peer.id,
                existing_peer.ip
            );
            return Ok(existing_peer);
        }

        match self.relay_addr {
            Some(relay_addr) => self.long_poll_relay(relay_addr).await,
            None => {
                log::debug!("Node {} waiting for peer introduction packet...", self.id);
                let intro = self.listen_introduction().await?;
                let details = NodeDetails::from(intro);
                self.coordinator.register_node(details.clone());
                log::info!(
                    "Node {} received and registered peer {} at {} (TCP: {:?})",
                    self.id,
                    details.id,
                    details.ip,
                    details.tcp_port
                );
                Ok(details)
            }
        }
    }

    /// Registers with a relay server and polls until a peer node is introduced.
    /// If any peer already exists in the coordinator, returns it immediately.
    pub async fn register_and_wait_for_peer(&mut self, relay_addr: SocketAddr) -> Result<NodeDetails, StruumError> {
        if let Some(existing_peer) = self.coordinator.select_node() {
            return Ok(existing_peer);
        }
        self.long_poll_relay(relay_addr).await
    }

    /// Returns the NodeDetails describing this node.
    pub fn details(&self) -> NodeDetails {
        NodeDetails {
            id: self.id,
            ip: SocketAddr::new(self.net.ip, self.net.udp_port),
            role: NodeRole::NODE,
            backend: Some(self.backend),
            tcp_port: self.advertise_tcp.then_some(self.net.tcp_port),
            quic_port: self.quic_port(),
        }
    }

    /// Converts this node into its NodeDetails representation.
    pub fn into_node_details(self) -> NodeDetails {
        self.details()
    }

    /// Registers a peer node in this node's embedded coordinator map.
    pub fn register_node(&mut self, details: NodeDetails) {
        self.coordinator.register_node(details);
    }

    /// Removes a node from this node's embedded coordinator map.
    pub fn deregister_node(&mut self, id: &UID) {
        self.coordinator.deregister_node(id);
    }

    /// Returns a reference to a registered peer node by UID.
    pub fn get_node(&self, id: &UID) -> Option<&NodeDetails> {
        self.coordinator.get_node(id)
    }

    /// Returns all registered peer nodes in this node's network map.
    pub fn get_nodes(&self) -> Vec<NodeDetails> {
        self.coordinator.get_nodes()
    }

    /// Number of registered peer nodes in this node's network map.
    pub fn node_count(&self) -> usize {
        self.coordinator.node_count()
    }

    /// Round-robin pick among registered peer nodes.
    pub fn select_node(&mut self) -> Option<NodeDetails> {
        self.coordinator.select_node()
    }

    /// Selects a registered peer node matching the requested computation backend.
    pub fn select_node_by_backend(&mut self, backend: NodeBackend) -> Option<NodeDetails> {
        self.coordinator.select_node_by_backend(backend)
    }

    /// Listens for a node introduction packet via UDP and registers it in this node's embedded coordinator map.
    pub async fn listen_for_node_registration(&mut self) -> Result<NodeDetails, StruumError> {
        let (intro, _src) = self.net.listen_udp::<IntroductionPacket>().await?;
        let details = NodeDetails::from(intro);
        self.register_node(details.clone());
        Ok(details)
    }

    /// Registers this node with a remote relay server by sending an Introduction packet.
    /// If peer IPs already exist in the coordinator, relay registration is skipped.
    pub async fn register_to_relay(&mut self, relay_addr: SocketAddr) -> Result<(), StruumError> {
        if self.coordinator.node_count() > 0 {
            log::info!(
                "Node {} already has {} peer(s) in coordinator. Skipping relay registration.",
                self.id,
                self.coordinator.node_count()
            );
            return Ok(());
        }
        self.relay_addr = Some(relay_addr);
        log::info!("Node {} sending registration packet to relay at {}...", self.id, relay_addr);
        self.introduce(relay_addr).await
    }

    /// Introduces two sibling peer nodes to each other so they can establish direct communication.
    pub async fn introduce_nodes(&mut self, node_a_id: &UID, node_b_id: &UID) -> Result<(), StruumError> {
        log::info!("Node {} introducing sibling nodes: {} <---> {}", self.id, node_a_id, node_b_id);
        let node_a = self
            .coordinator
            .get_node(node_a_id)
            .cloned()
            .ok_or_else(|| StruumError::NotFound(format!("Node {} not found", node_a_id)))?;
        let node_b = self
            .coordinator
            .get_node(node_b_id)
            .cloned()
            .ok_or_else(|| StruumError::NotFound(format!("Node {} not found", node_b_id)))?;

        // Send Node B's details to Node A
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    id: node_b.id,
                    ip: node_b.ip,
                    role: node_b.role,
                    backend: node_b.backend,
                    tcp_port: node_b.tcp_port,
                    quic_port: node_b.quic_port,
                }),
                node_a.ip,
            )
            .await?;

        // Send Node A's details to Node B
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    id: node_a.id,
                    ip: node_a.ip,
                    role: node_a.role,
                    backend: node_a.backend,
                    tcp_port: node_a.tcp_port,
                    quic_port: node_a.quic_port,
                }),
                node_b.ip,
            )
            .await?;

        Ok(())
    }
}

impl<const BUF_SIZE: usize> From<&Node<BUF_SIZE>> for NodeDetails {
    fn from(node: &Node<BUF_SIZE>) -> Self {
        node.details()
    }
}

impl<const BUF_SIZE: usize> From<Node<BUF_SIZE>> for NodeDetails {
    fn from(node: Node<BUF_SIZE>) -> Self {
        node.details()
    }
}

#[async_trait]
impl<const BUF_SIZE: usize> NetworkCommunicator for Node<BUF_SIZE> {
    /// LAN Broadcast introducing itself to the network.
    async fn notify_network(&mut self) -> Result<(), StruumError> {
        self.net.broadcast(UdpPacket::HELLO(HelloPacket {})).await
    }

    /// Sends a unicast UDP IntroductionPacket introducing itself to the recipient.
    async fn introduce(&mut self, ip: SocketAddr) -> Result<(), StruumError> {
        self.net
            .send_udp_data(
                UdpPacket::INTRODUCTION(IntroductionPacket {
                    id: self.id,
                    ip: SocketAddr::new(self.net.ip, self.net.udp_port),
                    role: NodeRole::NODE,
                    backend: Some(self.backend),
                    tcp_port: self.advertise_tcp.then_some(self.net.tcp_port),
                    quic_port: self.quic_port(),
                }),
                ip,
            )
            .await
    }

    /// Listens for a hello packet from the network.
    async fn listen_hello(&mut self) -> Result<SocketAddr, StruumError> {
        let (_, src) = self.net.listen_udp::<HelloPacket>().await?;
        log::info!("Node {} received Hello packet from {}", self.id, src);
        Ok(src)
    }

    /// Listens for an introduction packet from a coordinator or sibling node.
    async fn listen_introduction(&mut self) -> Result<IntroductionPacket, StruumError> {
        let (resp, _src) = self.net.listen_udp::<IntroductionPacket>().await?;
        log::info!(
            "Node {} received Introduction packet from sibling node {} at {}",
            self.id,
            resp.id,
            resp.ip
        );
        Ok(resp)
    }
}

#[async_trait]
impl<const BUF_SIZE: usize> NetworkCoordinator for Node<BUF_SIZE> {
    /// Looks up a registered node by ID.
    async fn introduce_sibling_node(&mut self, node_id: &UID) -> Result<&NodeDetails, StruumError> {
        self.coordinator.get_node(node_id).ok_or_else(|| {
            StruumError::NotFound("Node not found in registered network!".to_string())
        })
    }

    /// Introduces two sibling nodes to each other so they can establish direct communication.
    async fn introduce_nodes(&mut self, node_a_id: &UID, node_b_id: &UID) -> Result<(), StruumError> {
        self.introduce_nodes(node_a_id, node_b_id).await
    }

    /// Cross-introduces all registered sibling nodes to each other in the network (mesh coordination).
    async fn establish_information_exchange(&mut self) -> Result<(), StruumError> {
        let node_list: Vec<NodeDetails> = self.coordinator.get_nodes();
        for i in 0..node_list.len() {
            for j in (i + 1)..node_list.len() {
                let a = &node_list[i];
                let b = &node_list[j];
                self.introduce_nodes(&a.id, &b.id).await?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use struum_macros::gpu_type;
    use struum_relay::RelayServer;
    use struum_types::Buffer;

    #[gpu_type]
    #[derive(Debug, PartialEq)]
    struct Vec2 {
        x: f32,
        y: f32,
    }

    #[gpu_type]
    #[derive(Debug, PartialEq)]
    struct Param {
        m: f32,
        c: f32,
    }

    static GL_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[tokio::test]
    async fn test_node_intra_node_thread_scheduling() {
        let _lock = GL_TEST_MUTEX.lock().unwrap();
        let mut node = Node::<1024>::with_id(UID::new(10), 0, 0, NodeBackend::OpenGL, 2)
            .await
            .expect("Failed to initialize Node");

        assert_eq!(node.id, UID::new(10));
        assert_eq!(node.backend, NodeBackend::OpenGL);

        let source = r#"
            void compute(uint i) {
                Param p = param[0];
                Vec2 x = pos[i];
                output[i] = Vec2(
                    p.m * x.x,
                    p.c * x.y
                );
            }
        "#;

        let mut kernel = Kernel::new(source, "compute");
        kernel.add_buffer::<Param>(Buffer::new("param", &[Param { m: 3.0, c: 4.0 }]));
        kernel.add_buffer::<Vec2>(Buffer::new(
            "pos",
            &[Vec2 { x: 2.0, y: 5.0 }, Vec2 { x: 4.0, y: 10.0 }],
        ));
        kernel.add_buffer::<Vec2>(Buffer::empty::<Vec2>("output", 2));
        kernel.set_work_buffer("pos");
        kernel.pack().unwrap();

        // Submit job to local worker threads via struum_scheduler
        let job_id = node.submit_job(kernel).await;
        assert!(!job_id.is_empty());

        // Await completion notification from worker threads
        let completed = node.recv_completed_job().await.expect("Expected completed job ID");
        assert_eq!(completed, job_id);

        // Fetch result kernel
        let result = node.get_job_result(&completed).await.expect("Expected result kernel");
        let output = result.get_buffer("output").expect("Expected output buffer");
        let slice = output.as_slice::<Vec2>();

        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Vec2 { x: 6.0, y: 20.0 });
        assert_eq!(slice[1], Vec2 { x: 12.0, y: 40.0 });
    }

    #[tokio::test]
    async fn test_node_to_node_details_conversion() {
        let node = Node::<1024>::new(0, 0, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node");

        // Test details()
        let details = node.details();
        assert_eq!(details.id, node.id);
        assert_eq!(details.ip.port(), node.net.udp_port);
        assert_eq!(details.role, NodeRole::NODE);
        assert_eq!(details.backend, Some(NodeBackend::Cpu));
        assert_eq!(details.tcp_port, Some(node.net.tcp_port));

        // Test From<&Node>
        let from_ref = NodeDetails::from(&node);
        assert_eq!(from_ref, details);

        // Test into_node_details()
        let into_details = node.into_node_details();
        assert_eq!(into_details, details);
    }

    #[tokio::test]
    async fn test_node_embedded_coordinator() {
        let mut node = Node::<1024>::new(0, 0, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node");

        let peer1 = NodeDetails {
            id: UID::new(101),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38341),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::OpenGL),
            tcp_port: Some(38342),
            quic_port: None,
        };
        let peer2 = NodeDetails {
            id: UID::new(102),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38351),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
            tcp_port: Some(38352),
            quic_port: None,
        };

        // Access via node delegation
        node.register_node(peer1.clone());
        // Access via embedded coordinator field directly
        node.coordinator.register_node(peer2.clone());

        assert_eq!(node.node_count(), 2);
        assert_eq!(node.coordinator.node_count(), 2);
        assert_eq!(node.get_node(&UID::new(101)), Some(&peer1));
        assert_eq!(node.coordinator.get_node(&UID::new(102)), Some(&peer2));

        let selected = node.select_node();
        assert!(selected.is_some());

        let opengl_peer = node.select_node_by_backend(NodeBackend::OpenGL);
        assert_eq!(opengl_peer.map(|p| p.id), Some(UID::new(101)));

        node.deregister_node(&UID::new(101));
        assert_eq!(node.get_node(&UID::new(101)), None);
        assert_eq!(node.node_count(), 1);
    }

    #[tokio::test]
    async fn test_tcp_task_data_transfer() {
        let _lock = GL_TEST_MUTEX.lock().unwrap();
        // Node 1 (Sender / Requester)
        let mut node1 = Node::<1024>::with_id(UID::new(1), 0, 0, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node 1");

        // Node 2 (Receiver / Worker)
        let mut node2 = Node::<1024>::with_id(UID::new(2), 0, 0, NodeBackend::OpenGL, 2)
            .await
            .expect("Failed to initialize Node 2");

        let node2_tcp_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), node2.net.tcp_port);

        // Spawn receiver task on Node 2
        let receiver_task = tokio::spawn(async move {
            let conn_peer = node2.accept_peer_connection().await.unwrap();
            let (task_id, sender_id, kernel) = node2.recv_kernel(&conn_peer).await.unwrap();
            assert_eq!(task_id, "job-test-42");
            assert_eq!(sender_id, UID::new(1));

            // Verify that bindings and source transferred faithfully
            let pos_buf = kernel.get_buffer("pos").expect("Expected pos buffer");
            let pos_slice = pos_buf.as_slice::<Vec2>();
            assert_eq!(pos_slice.len(), 2);
            assert_eq!(pos_slice[0], Vec2 { x: 2.0, y: 5.0 });

            // Execute the computation remotely using node2's worker threads
            let local_job_id = node2.submit_job(kernel).await;
            let completed = node2.recv_completed_job().await.expect("Expected completed job ID");
            assert_eq!(completed, local_job_id);

            let result_kernel = node2.get_job_result(&completed).await.expect("Expected result kernel");

            // Send result back to requester over TCP
            node2.send_task_result(conn_peer, &task_id, &result_kernel).await.unwrap();
        });

        // Sender connects to Node 2 over TCP
        node1.connect_to_peer_addr(node2_tcp_addr).await.expect("Failed to connect to peer TCP");

        // Prepare computation kernel with source and buffer bindings
        let source = r#"
            void compute(uint i) {
                Param p = param[0];
                Vec2 x = pos[i];
                output[i] = Vec2(
                    p.m * x.x,
                    p.c * x.y
                );
            }
        "#;

        let mut kernel = Kernel::new(source, "compute");
        kernel.add_buffer::<Param>(Buffer::new("param", &[Param { m: 3.0, c: 4.0 }]));
        kernel.add_buffer::<Vec2>(Buffer::new(
            "pos",
            &[Vec2 { x: 2.0, y: 5.0 }, Vec2 { x: 4.0, y: 10.0 }],
        ));
        kernel.add_buffer::<Vec2>(Buffer::empty::<Vec2>("output", 2));
        kernel.set_work_buffer("pos");
        kernel.pack().unwrap();

        // Node 1 sends kernel to Node 2 over TCP
        node1.send_kernel(node2_tcp_addr, "job-test-42", &kernel)
            .await
            .expect("Failed to send kernel over TCP");

        // Node 1 receives computation result from Node 2 over TCP
        let (job_id, sender_id, result_kernel) = node1.recv_task_result(&node2_tcp_addr)
            .await
            .expect("Failed to receive task result over TCP");

        assert_eq!(job_id, "job-test-42");
        assert_eq!(sender_id, UID::new(2));

        let output = result_kernel.get_buffer("output").expect("Expected output buffer");
        let slice = output.as_slice::<Vec2>();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0], Vec2 { x: 6.0, y: 20.0 });
        assert_eq!(slice[1], Vec2 { x: 12.0, y: 40.0 });

        receiver_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_delayed_relay_startup_polling() {
        let _lock = GL_TEST_MUTEX.lock().unwrap();

        // A port that is free now; the relay binds it later in the test.
        let relay_port = std::net::UdpSocket::bind("0.0.0.0:0").unwrap().local_addr().unwrap().port();
        let relay_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), relay_port);

        // Node 1 & Node 2 start BEFORE the relay server is created or running
        let mut node1 = Node::<1024>::with_id(UID::new(1), 0, 0, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node 1");
        let n1_tcp = node1.net.tcp_port;
        let mut node2 = Node::<1024>::with_id(UID::new(2), 0, 0, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node 2");
        let n2_tcp = node2.net.tcp_port;

        // Spawn async tasks for both nodes long-polling the offline relay
        let n1_handle = tokio::spawn(async move {
            node1.long_poll_relay(relay_addr).await
        });
        let n2_handle = tokio::spawn(async move {
            node2.long_poll_relay(relay_addr).await
        });

        // Simulate relay downtime: wait 1.5 seconds while nodes are long-polling
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

        // Verify neither task has completed yet since relay was offline
        assert!(!n1_handle.is_finished());
        assert!(!n2_handle.is_finished());

        // NOW start the relay server
        let mut relay = RelayServer::<1024>::new(relay_port, 0)
            .await
            .expect("Failed to create RelayServer");

        let relay_handle = tokio::spawn(async move {
            let _ = relay.run().await;
        });

        // Await introductions with a timeout
        let n1_peer = tokio::time::timeout(std::time::Duration::from_secs(5), n1_handle)
            .await
            .expect("Timed out waiting for Node 1 peer introduction")
            .unwrap()
            .expect("Node 1 failed to receive peer introduction");

        let n2_peer = tokio::time::timeout(std::time::Duration::from_secs(5), n2_handle)
            .await
            .expect("Timed out waiting for Node 2 peer introduction")
            .unwrap()
            .expect("Node 2 failed to receive peer introduction");

        // Verify Node 1 discovered Node 2 and vice versa
        assert_eq!(n1_peer.id, UID::new(2));
        assert_eq!(n1_peer.tcp_port, Some(n2_tcp));

        assert_eq!(n2_peer.id, UID::new(1));
        assert_eq!(n2_peer.tcp_port, Some(n1_tcp));

        relay_handle.abort();
    }

    #[tokio::test]
    async fn test_skip_relay_if_peer_exists_in_coordinator() {
        let mut node = Node::<1024>::with_id(UID::new(10), 0, 0, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node");

        let existing_peer = NodeDetails {
            id: UID::new(20),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38821),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
            tcp_port: Some(38822),
            quic_port: None,
        };

        // Manually insert known peer into node's coordinator
        node.register_node(existing_peer.clone());
        assert_eq!(node.node_count(), 1);

        // Dummy offline relay address
        let bogus_relay_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 39999);

        // 1. long_poll_relay should immediately return existing peer without calling relay
        let peer = node.long_poll_relay(bogus_relay_addr).await.expect("Expected immediate peer");
        assert_eq!(peer.id, UID::new(20));
        assert_eq!(peer.tcp_port, Some(38822));

        // 2. register_to_relay should immediately succeed without sending to relay
        node.register_to_relay(bogus_relay_addr).await.expect("Expected immediate success");

        // 3. listen_sibling_introduction should immediately return existing peer
        let peer2 = node.listen_sibling_introduction().await.expect("Expected immediate peer");
        assert_eq!(peer2.id, UID::new(20));

        // 4. long_poll_relay_for_peers(1) should immediately return existing peer list
        let peers = node.long_poll_relay_for_peers(bogus_relay_addr, 1).await.expect("Expected immediate peers");
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].id, UID::new(20));
    }
}



/// Serves requesters on behalf of a [`Node`]; see [`Node::server`].
#[derive(Clone)]
pub struct PeerServer {
    id: UID,
    scheduler: JobScheduler,
    stats: Arc<NodeStats>,
    blob_cache_bytes: usize,
}

impl PeerServer {
    /// Serves one requester: executes every `WORK` / `WORKREF` (and legacy
    /// `TASKDATA`) packet it sends, in order, and replies with `RESULT`,
    /// `WORKERROR` or `CACHEMISS`. Returns when the requester disconnects.
    /// Several connections can be served at once; they share the node's workers.
    pub async fn serve(self, peer: PeerConnection) {
        let peer_addr = peer.addr;
        // Blobs this requester sent us, so later `WORKREF`s can leave them out.
        let mut blobs = BlobStore::new(self.blob_cache_bytes);
        loop {
            let packet = match peer.recv().await {
                Ok(p) => p,
                Err(e) => {
                    log::info!("Node {} stopped serving {}: {}", self.id, peer_addr, e);
                    return;
                }
            };

            // `elide`: WORK results leave out buffers the task did not change (the
            // requester restores them); legacy TASKDATA results are always complete.
            let (task_id, kind, work, elide, bytes_in) = match packet {
                TcpPacket::WORK(WorkPacket { task_id, kind, data, .. }) => {
                    let work = Work::from_bytes(kind, &data);
                    if let Ok(w) = &work {
                        blobs.remember(w);
                    }
                    (task_id, kind, work, true, data.len())
                }
                TcpPacket::WORKREF(WorkRefPacket { task_id, kind, refs, data, .. }) => {
                    let work = Work::from_bytes(kind, &data).and_then(|mut w| {
                        blobs.remember(&w); // blobs sent inline alongside the refs
                        Ok(blobs.resolve(&mut w, &refs).map(|()| w))
                    });
                    let work = match work {
                        Ok(Ok(w)) => Ok(w),
                        Ok(Err(missing)) => {
                            log::debug!("Node {} lacks {} blob(s) for '{}'", self.id, missing.len(), task_id);
                            let miss = TcpPacket::CACHEMISS(CacheMissPacket { task_id, missing });
                            if let Err(e) = peer.send(&miss).await {
                                log::info!("Node {} stopped serving {}: {}", self.id, peer_addr, e);
                                return;
                            }
                            continue;
                        }
                        Err(e) => Err(e),
                    };
                    (task_id, kind, work, true, data.len())
                }
                TcpPacket::TASKDATA(TaskDataPacket { task_id, kernel_data, .. }) => {
                    let kind = struum_types::network::WorkKind::Gpu;
                    (task_id, kind, Work::from_bytes(kind, &kernel_data), false, kernel_data.len())
                }
                other => {
                    log::warn!("Node {} ignoring unexpected packet from {}: {:?}", self.id, peer_addr, other);
                    continue;
                }
            };

            log::info!("Node {} running {:?} task '{}' from {}", self.id, kind, task_id, peer_addr);
            let started = std::time::Instant::now();
            self.stats.begin_serving();
            let result = match work {
                Ok(work) => {
                    let original = elide.then(|| work.clone());
                    self.scheduler.run(work).await.map(|mut done| {
                        if let Some(original) = &original {
                            done.elide_unchanged(original);
                        }
                        done
                    })
                }
                Err(e) => Err(e),
            };

            let busy = started.elapsed();
            let reply = match result.and_then(|w| w.to_bytes()) {
                Ok(bytes) => {
                    self.stats.served(peer_addr, true, bytes_in as u64, bytes.len() as u64, busy);
                    log::info!("Node {} finished '{}' in {:?}", self.id, task_id, started.elapsed());
                    TcpPacket::RESULT(ResultPacket::new(task_id, self.id, bytes))
                }
                Err(e) => {
                    self.stats.served(peer_addr, false, bytes_in as u64, 0, busy);
                    log::error!("Node {} failed '{}': {}", self.id, task_id, e);
                    TcpPacket::WORKERROR(WorkErrorPacket {
                        task_id,
                        sender_id: self.id,
                        message: e.to_string(),
                    })
                }
            };
            if let Err(e) = peer.send(&reply).await {
                log::info!("Node {} stopped serving {}: {}", self.id, peer_addr, e);
                return;
            }
        }
    }
}
