//! `struum-tui`: join a struum network as a worker and watch it live.
//!
//! ```text
//! struum-tui [--relay ADDR] [--backend cpu|opengl] [--udp-port N] [--quic-port N] [--tcp-port N]
//! ```
//! Env equivalents: `RELAY_ADDR` (default `127.0.0.1:39001`), `WORKER_BACKEND`,
//! `NODE_UDP_PORT`, `NODE_QUIC_PORT`, `NODE_TCP_PORT` (default 0: OS-assigned),
//! `RUST_LOG` (level). `STRUUM_TOKEN` (env only) is the shared secret requesters
//! must know; with it set, TCP is off and only QUIC is served.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use struum_node::Node;
use struum_tui::app::{AppState, History, NodeInfo, SortKey, neighbour_rows};
use struum_tui::discovery::{self, Discovery, Registration};
use struum_node::serving;
use struum_tui::{logger, selftest, ui};
use struum_types::StruumError;
use struum_types::network::{NodeBackend, UID};

const TICK: Duration = Duration::from_millis(250);
const REGISTER_EVERY: Duration = Duration::from_secs(2);

#[derive(Debug)]
struct Config {
    relay: SocketAddr,
    backend: Option<String>,
    udp_port: u16,
    quic_port: u16,
    tcp_port: u16,
}

const USAGE: &str =
    "usage: struum-tui [--relay ADDR] [--backend cpu|opengl] [--udp-port N] [--quic-port N] [--tcp-port N]";

fn config() -> Result<Config, String> {
    let env = |k: &str| std::env::var(k).ok();
    let mut relay = env("RELAY_ADDR").unwrap_or_else(|| "127.0.0.1:39001".into());
    let mut backend = env("WORKER_BACKEND");
    let mut udp = env("NODE_UDP_PORT").unwrap_or_else(|| "0".into());
    let mut quic = env("NODE_QUIC_PORT").unwrap_or_else(|| "0".into());
    let mut tcp = env("NODE_TCP_PORT").unwrap_or_else(|| "0".into());
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value\n{USAGE}"));
        match flag.as_str() {
            "--relay" => relay = value()?,
            "--backend" => backend = Some(value()?),
            "--udp-port" => udp = value()?,
            "--quic-port" => quic = value()?,
            "--tcp-port" => tcp = value()?,
            "-h" | "--help" => return Err(USAGE.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    Ok(Config {
        relay: relay.parse().map_err(|e| format!("bad relay address {relay}: {e}"))?,
        backend,
        udp_port: udp.parse().map_err(|e| format!("bad UDP port {udp}: {e}"))?,
        quic_port: quic.parse().map_err(|e| format!("bad QUIC port {quic}: {e}"))?,
        tcp_port: tcp.parse().map_err(|e| format!("bad TCP port {tcp}: {e}"))?,
    })
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    let cfg = match config() {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };
    let logs = logger::init(500);

    // The node serves QUIC (and TCP without a token); discovery uses its own UDP socket.
    let mut node = Node::<2048>::with_id(UID::generate(), 0, cfg.tcp_port, NodeBackend::Cpu, 1).await?;
    println!("struum-tui: running self-tests…");
    node.backend = selftest::choose_backend(&mut node, cfg.backend.as_deref()).await?;
    let (socket, udp_port) = discovery::bind(cfg.udp_port)
        .await
        .map_err(|e| StruumError::NetworkConnectionError(format!("UDP port {}: {e}", cfg.udp_port)))?;

    let (id, ip, backend) = (node.id, node.net.ip, node.backend);
    let stats = node.stats();
    let token = struum_network::quic::token_from_env();
    let listening = serving::start(node, cfg.quic_port, &token)?;
    let info = NodeInfo {
        id,
        ip,
        udp_port,
        quic_port: listening.quic_port,
        tcp_port: listening.tcp_port,
        authenticated: listening.authenticated,
        backend,
    };
    let shared = Arc::new(Mutex::new(Discovery::new(cfg.relay)));
    log::info!("Node {} joining via relay {} ({:?})", info.id, cfg.relay, info.backend);

    let reg = Registration {
        id,
        ip,
        backend,
        tcp_port: listening.tcp_port,
        quic_port: Some(listening.quic_port),
    };
    tokio::spawn(discovery::run(socket, reg, shared.clone(), REGISTER_EVERY));

    // The terminal loop blocks on input polling, so it gets its own thread.
    let ui = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
        let mut terminal = ratatui::init(); // raw mode + alt screen + panic hook that restores
        let mut history = History::default();
        let mut sort = SortKey::default();
        let mut show_help = false;
        let result = loop {
            let snapshot = stats.snapshot();
            history.sample(Instant::now(), &snapshot.served, Duration::from_secs(1));
            let state = {
                let d = shared.lock().unwrap();
                AppState {
                    node: info.clone(),
                    relay: d.relay,
                    relay_status: d.status.clone(),
                    peers: neighbour_rows(&d, &snapshot, sort),
                    stats: snapshot,
                    shards_per_sec: history.shards.iter().copied().collect(),
                    bytes_per_sec: history.bytes.iter().copied().collect(),
                    logs: logs.tail(200),
                    sort,
                    show_help,
                }
            };
            if let Err(e) = terminal.draw(|f| ui::render(f, &state)) {
                break Err(e);
            }
            match event::poll(TICK) {
                Ok(true) => match event::read() {
                    Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => match k.code {
                        KeyCode::Char('q') | KeyCode::Esc => break Ok(()),
                        KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => break Ok(()),
                        KeyCode::Tab => sort = sort.next(),
                        KeyCode::Char('?') => show_help = !show_help,
                        _ => {}
                    },
                    Ok(_) => {}
                    Err(e) => break Err(e),
                },
                Ok(false) => {}
                Err(e) => break Err(e),
            }
        };
        ratatui::restore();
        result
    });
    match ui.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => eprintln!("terminal error: {e}"),
        Err(e) => eprintln!("UI thread panicked: {e}"),
    }
    // Background tasks are dropped with the runtime.
    std::process::exit(0);
}
