//! Terminal dashboard for a struum worker.
//!
//! `struum-tui` joins a network through a relay, serves shards for any requester
//! that connects (like `struum_worker`), and shows live what this node contributed,
//! what its neighbours did for it, and who is on the network.
//!
//! The pieces are kept separate so the screen can be tested without a terminal:
//! - [`discovery`]: registers with the relay on its own UDP socket and keeps the
//!   list of introduced peers up to date while the node is busy serving.
//! - [`app`]: [`app::AppState`], a plain struct built from a
//!   [`StatsSnapshot`](struum_node::StatsSnapshot) and the peer list.
//! - [`ui`]: draws an [`app::AppState`] with ratatui.
//! - [`logger`]: a `log` backend writing into a ring buffer shown in the log pane.

pub mod app;
pub mod discovery;
pub mod logger;
pub mod selftest;
pub mod ui;
