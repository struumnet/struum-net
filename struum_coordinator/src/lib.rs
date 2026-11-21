//! A coordinator role on top of [`struum_node::Node`].
//!
//! [`Coordinator`] wraps a CPU [`Node`](struum_node::Node) (and derefs to it) and implements
//! [`NetworkCoordinator`](struum_types::network::NetworkCoordinator): it keeps track of
//! registered nodes and introduces them to each other. Most deployments use the
//! standalone relay (`struum_relay`) for discovery instead; a coordinator is useful when
//! one node should do both.

pub mod coordinator;

pub use coordinator::Coordinator;
