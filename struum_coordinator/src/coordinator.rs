use struum_types::{StruumError, network::{NetworkCoordinator, NodeDetails,UID}};
use std::collections::HashMap;
use struum_network::layer::NetworkLayer;
use async_trait::async_trait;

/// Represents the Coordinator for connections
pub struct Coordinator<const BUF_SIZE: usize> {
    _id: UID,
    net: NetworkLayer::<BUF_SIZE>,
    nodes: HashMap<UID,NodeDetails>,
}

impl <const BUF_SIZE:usize> Coordinator<BUF_SIZE>{
    pub async fn new(udp_port: u16, tcp_port: u16)->Result<Self,StruumError>{
        Ok(Self{
            _id: UID::new(0),
            net: NetworkLayer::new(udp_port,tcp_port).await?,
            nodes: HashMap::default(),
        })
    }
}

#[async_trait]
impl<const BUF_SIZE: usize> NetworkCoordinator for  Coordinator<BUF_SIZE> {
    async fn introduce_sibling_node(&mut self,node_id: &UID) -> Result<&NodeDetails, StruumError>{
        self.nodes.get(node_id).ok_or(StruumError::NotFound("Node not found in registered network!".to_string()))
    }
    async fn establish_information_exchange(&mut self)->Result<(),StruumError>{
        Ok(())
    }
}
