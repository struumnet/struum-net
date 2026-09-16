use std::collections::HashMap;
use std::net::SocketAddr;
use async_trait::async_trait;
use tokio::sync::mpsc::{self, Receiver};

use struum_kernel::Kernel;
use struum_network::layer::NetworkLayer;
use struum_scheduler::{JobId, JobScheduler};
use struum_types::StruumError;
use struum_types::network::{
    HelloPacket, IntroductionPacket, NetworkCommunicator, NetworkCoordinator, NodeBackend,
    NodeDetails, NodeRole, ResultPacket, TaskDataPacket, TaskPacket, TcpPacket, UID, UdpPacket,
};

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
}

impl<const BUF_SIZE: usize> std::fmt::Debug for Node<BUF_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("net", &self.net)
            .field("backend", &self.backend)
            .field("peers_count", &self.coordinator.node_count())
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
        })
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

    /// Listens for an introduction packet sent by the coordinator or relay introducing a sibling node.
    pub async fn listen_sibling_introduction(&mut self) -> Result<NodeDetails, StruumError> {
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

    /// Returns the NodeDetails describing this node.
    pub fn details(&self) -> NodeDetails {
        NodeDetails {
            id: self.id,
            ip: SocketAddr::new(self.net.ip, self.net.udp_port),
            role: NodeRole::NODE,
            backend: Some(self.backend),
            tcp_port: Some(self.net.tcp_port),
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
    pub async fn register_to_relay(&mut self, relay_addr: SocketAddr) -> Result<(), StruumError> {
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
                    tcp_port: Some(self.net.tcp_port),
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
        let mut node = Node::<1024>::with_id(UID::new(10), 38111, 38112, NodeBackend::OpenGL, 2)
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
        let node = Node::<1024>::new(38221, 38222, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node");

        // Test details()
        let details = node.details();
        assert_eq!(details.id, node.id);
        assert_eq!(details.ip.port(), 38221);
        assert_eq!(details.role, NodeRole::NODE);
        assert_eq!(details.backend, Some(NodeBackend::Cpu));
        assert_eq!(details.tcp_port, Some(38222));

        // Test From<&Node>
        let from_ref = NodeDetails::from(&node);
        assert_eq!(from_ref, details);

        // Test into_node_details()
        let into_details = node.into_node_details();
        assert_eq!(into_details, details);
    }

    #[tokio::test]
    async fn test_node_embedded_coordinator() {
        let mut node = Node::<1024>::new(38331, 38332, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node");

        let peer1 = NodeDetails {
            id: UID::new(101),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38341),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::OpenGL),
            tcp_port: Some(38342),
        };
        let peer2 = NodeDetails {
            id: UID::new(102),
            ip: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 38351),
            role: NodeRole::NODE,
            backend: Some(NodeBackend::Cpu),
            tcp_port: Some(38352),
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
        let mut node1 = Node::<1024>::with_id(UID::new(1), 38411, 38412, NodeBackend::Cpu, 1)
            .await
            .expect("Failed to initialize Node 1");

        // Node 2 (Receiver / Worker)
        let mut node2 = Node::<1024>::with_id(UID::new(2), 38421, 38422, NodeBackend::OpenGL, 2)
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
}
