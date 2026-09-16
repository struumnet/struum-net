use std::net::SocketAddr;
use async_trait::async_trait;
use tokio::sync::mpsc::{self, Receiver};

use struum_kernel::Kernel;
use struum_network::layer::NetworkLayer;
use struum_scheduler::{JobId, JobScheduler};
use struum_types::StruumError;
use struum_types::network::{
    HelloPacket, IntroductionPacket, NetworkCommunicator, NodeBackend, NodeDetails, NodeRole,
    TaskPacket, TcpPacket, UID, UdpPacket,
};

/// Represents a computational Node in the network.
/// Inter-node communication is coordinated via `NetworkCommunicator` / `struum_coordinator`,
/// while intra-node worker threads communicate and execute jobs via `struum_scheduler`.
pub struct Node<const BUF_SIZE: usize> {
    /// Id is immutable and uniquely identifies the node.
    pub id: UID,
    /// Network layer with BUF_SIZE buffer for network communication.
    pub net: NetworkLayer<BUF_SIZE>,
    /// Defines the computation backend (e.g. CPU, OpenGL).
    pub backend: NodeBackend,
    /// Intra-node scheduler communicating and scheduling tasks across local worker threads.
    pub scheduler: JobScheduler,
    /// Channel receiver for completed job IDs from local worker threads.
    pub job_rx: Receiver<JobId>,
}

impl<const BUF_SIZE: usize> std::fmt::Debug for Node<BUF_SIZE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("net", &self.net)
            .field("backend", &self.backend)
            .finish()
    }
}

impl<const BUF_SIZE: usize> Node<BUF_SIZE> {
    /// Creates a new Node with a specified number of local worker threads.
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

        Ok(Self {
            id,
            net,
            backend,
            scheduler,
            job_rx,
        })
    }

    // --- Intra-node thread communication methods (via struum_scheduler) ---

    /// Submits a kernel to be scheduled and executed by worker threads on this node.
    pub async fn submit_job(&self, kernel: Kernel) -> JobId {
        let id = self.scheduler.add_job(kernel).await;
        log::info!("Node {} submitted job {} to local scheduler", self.id, id);
        id
    }

    /// Waits for the next job completed by one of the local worker threads.
    pub async fn recv_completed_job(&mut self) -> Option<JobId> {
        let id = self.job_rx.recv().await;
        if let Some(ref job_id) = id {
            log::info!("Node {} received completion notice for job {}", self.id, job_id);
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

    /// Sends a computation task packet to a peer node over TCP.
    pub async fn send_task_to_peer(
        &mut self,
        peer_addr: SocketAddr,
        task: TaskPacket,
    ) -> Result<(), StruumError> {
        self.net.send_tcp_data(peer_addr, TcpPacket::TASK(task)).await
    }

    /// Listens for an introduction packet sent by the coordinator introducing a sibling node.
    pub async fn listen_sibling_introduction(&mut self) -> Result<NodeDetails, StruumError> {
        let intro = self.listen_introduction().await?;
        Ok(NodeDetails::from(intro))
    }

    /// Returns the NodeDetails describing this node.
    pub fn details(&self) -> NodeDetails {
        NodeDetails {
            id: self.id,
            ip: SocketAddr::new(self.net.ip, self.net.udp_port),
            role: NodeRole::NODE,
            backend: Some(self.backend),
        }
    }

    /// Converts this node into its NodeDetails representation.
    pub fn into_node_details(self) -> NodeDetails {
        self.details()
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

#[cfg(test)]
mod tests {
    use super::*;
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

    #[tokio::test]
    async fn test_node_intra_node_thread_scheduling() {
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

        // Test From<&Node>
        let from_ref = NodeDetails::from(&node);
        assert_eq!(from_ref, details);

        // Test into_node_details()
        let into_details = node.into_node_details();
        assert_eq!(into_details, details);
    }
}

