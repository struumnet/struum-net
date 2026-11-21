//! End to end without a terminal: relay + this node (discovery and serving, wired
//! as in main.rs) + a requester that finds us through the relay and runs a job.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use struum_cpu::selftest;
use struum_node::{Node, ShardOptions};
use struum_relay::RelayServer;
use struum_tui::app::{AppState, History, NodeInfo, SortKey, neighbour_rows};
use struum_tui::discovery::{self, Discovery, RelayStatus, Registration};
use struum_tui::ui;
use struum_types::network::{NodeBackend, UID};

fn render(state: &AppState) -> String {
    let (w, h) = (120, 34);
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| ui::render(f, state)).unwrap();
    let buf = term.backend().buffer().clone();
    (0..h)
        .map(|y| (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>().trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

#[tokio::test(flavor = "multi_thread")]
async fn joins_serves_and_shows_contributions() {
    // Relay.
    let mut relay = RelayServer::<2048>::new(0, 0).await.unwrap();
    let relay_addr = SocketAddr::from(([127, 0, 0, 1], relay.net.udp_port));
    tokio::spawn(async move { relay.run().await });

    // This node: serving Node + discovery socket.
    let mut node = Node::<2048>::with_id(UID::generate(), 0, 0, NodeBackend::Cpu, 1).await.unwrap();
    let info = NodeInfo {
        id: node.id,
        ip: node.net.ip,
        udp_port: 0,
        tcp_port: node.net.tcp_port,
        backend: NodeBackend::Cpu,
    };
    let stats = node.stats();
    let shared = Arc::new(Mutex::new(Discovery::new(relay_addr)));
    let (socket, _) = discovery::bind(0).await.unwrap();
    let reg = Registration { id: info.id, ip: info.ip, backend: info.backend, tcp_port: info.tcp_port };
    tokio::spawn(discovery::run(socket, reg, shared.clone(), Duration::from_millis(200)));
    tokio::spawn(async move {
        loop {
            let peer = node.accept_peer_connection().await.unwrap();
            node.serve_peer(peer).await.unwrap();
        }
    });

    // A requester elsewhere on the network.
    let mut requester = Node::<2048>::with_id(UID::generate(), 0, 0, NodeBackend::Cpu, 1).await.unwrap();
    let requester_id = requester.id;
    let peers = tokio::time::timeout(
        Duration::from_secs(10),
        requester.discover_peers(relay_addr, 1, Duration::from_millis(300)),
    )
    .await
    .expect("requester finds us through the relay")
    .unwrap();
    assert!(peers.iter().any(|p| p.id == info.id));
    assert_eq!(requester.connect_to_peers(&peers).await.len(), 1);
    let opts = ShardOptions { shards_per_peer: 4, ..ShardOptions::default() };
    let out = requester.run_sharded(selftest::task(2000), &opts).await.unwrap();
    assert_eq!(
        out.into_cpu_task().unwrap().get_buffer("y").unwrap().as_slice::<f32>(),
        selftest::expected(2000).as_slice()
    );

    // Our side learned about the requester through the relay.
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if shared.lock().unwrap().peers.contains_key(&requester_id) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("relay introduces the requester to us");

    let snapshot = stats.snapshot();
    assert_eq!(snapshot.served.shards, 4);
    let mut history = History::default();
    history.sample(std::time::Instant::now(), &snapshot.served, Duration::from_secs(1));
    let state = {
        let d = shared.lock().unwrap();
        assert_eq!(d.status, RelayStatus::Connected);
        AppState {
            node: info.clone(),
            relay: d.relay,
            relay_status: d.status.clone(),
            peers: neighbour_rows(&d, &snapshot, SortKey::ForThem),
            stats: snapshot,
            shards_per_sec: history.shards.iter().copied().collect(),
            bytes_per_sec: history.bytes.iter().copied().collect(),
            logs: vec![],
            sort: SortKey::ForThem,
            show_help: false,
        }
    };
    // Both advertise the same LAN IP here, but the requester is the only peer on it,
    // so its ephemeral source address is attributed to it.
    let row = state.peers.iter().find(|r| r.id == Some(requester_id)).expect("requester row");
    assert_eq!(row.for_them.shards, 4);

    let screen = render(&state);
    println!("{screen}");
    assert!(screen.contains("connected"), "{screen}");
    assert!(screen.contains("for others 4 shards  0 failed"), "{screen}");
    assert!(screen.contains(&ui::id(requester_id)), "{screen}");
}
