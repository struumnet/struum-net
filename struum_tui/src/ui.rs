//! Draws an [`AppState`] with ratatui.

use std::time::Duration;

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Clear, Paragraph, Row, Sparkline, Table, Wrap};

use crate::app::AppState;
use crate::discovery::RelayStatus;
use struum_node::Tally;
use struum_types::network::UID;

/// Node id as 16 hex digits (fits the neighbour table, unlike `UID`'s Debug-style Display).
pub fn id(uid: UID) -> String {
    format!("{:016x}", uid.inner)
}

/// Human-readable byte count.
pub fn mb(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2} GB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Short duration: `950ms`, `12.3s`, `4m05s`, `2h03m`.
pub fn dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, (s / 60) % 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else if d.as_millis() >= 1000 {
        format!("{:.1}s", d.as_secs_f64())
    } else {
        format!("{}ms", d.as_millis())
    }
}

fn title(text: &str) -> Block<'static> {
    Block::default().borders(Borders::ALL).title(Span::styled(
        format!(" {text} "),
        Style::default().add_modifier(Modifier::BOLD),
    ))
}

pub fn render(f: &mut Frame, s: &AppState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(9),
            Constraint::Min(6),
            Constraint::Length(8),
        ])
        .split(f.area());
    header(f, rows[0], s);
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);
    contribution(f, mid[0], s);
    network(f, mid[1], s);
    neighbours(f, rows[2], s);
    logs(f, rows[3], s);
    if s.show_help {
        help(f);
    }
}

fn header(f: &mut Frame, area: Rect, s: &AppState) {
    let (relay, color) = match &s.relay_status {
        RelayStatus::Registering => ("registering…".to_string(), Color::Yellow),
        RelayStatus::Connected => ("connected".to_string(), Color::Green),
        RelayStatus::Error(e) => (format!("error: {e}"), Color::Red),
    };
    let (state, state_color) = if s.stats.working { ("WORKING", Color::Green) } else { ("idle", Color::DarkGray) };
    let lines = vec![
        Line::from(vec![
            "node ".dim(),
            Span::raw(id(s.node.id)).bold(),
            "  ip ".dim(),
            Span::raw(s.node.ip.to_string()),
            "  udp ".dim(),
            Span::raw(s.node.udp_port.to_string()),
            "  tcp ".dim(),
            Span::raw(s.node.tcp_port.to_string()),
            "  backend ".dim(),
            Span::raw(format!("{:?}", s.node.backend)),
        ]),
        Line::from(vec![
            "relay ".dim(),
            Span::raw(s.relay.to_string()),
            " ".into(),
            Span::styled(relay, Style::default().fg(color)),
            "  up ".dim(),
            Span::raw(dur(s.stats.uptime)),
            "  ".into(),
            Span::styled(state, Style::default().fg(state_color).add_modifier(Modifier::BOLD)),
        ]),
    ];
    f.render_widget(Paragraph::new(lines).block(title("struum-tui · q quit · tab sort · ? help")), area);
}

fn tally_line(label: &str, t: &Tally) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<11}"), Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!(
            "{} shards  {} failed  in {}  out {}  busy {}",
            t.shards,
            t.failed,
            mb(t.bytes_in),
            mb(t.bytes_out),
            dur(t.busy)
        )),
    ])
}

fn contribution(f: &mut Frame, area: Rect, s: &AppState) {
    let block = title("Your contribution");
    let inner = block.inner(area);
    f.render_widget(block, area);
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(1)])
        .split(inner);
    let peak = s.shards_per_sec.iter().max().copied().unwrap_or(0);
    let last_bytes = s.bytes_per_sec.last().copied().unwrap_or(0);
    let text = vec![
        tally_line("for others", &s.stats.served),
        tally_line("for you", &s.stats.used),
        Line::from(vec![
            "utilisation ".dim(),
            Span::raw(format!("{:.1}%", s.utilisation() * 100.0)),
            "  requesters ".dim(),
            Span::raw(s.stats.served_by_peer.len().to_string()),
            "  now ".dim(),
            Span::raw(format!("{}/s", mb(last_bytes))),
        ]),
        Line::from(vec![
            "served shards/s, last 60s ".dim(),
            Span::raw(format!("(peak {peak})")),
        ]),
    ];
    f.render_widget(Paragraph::new(text), parts[0]);
    f.render_widget(
        Sparkline::default().data(&s.shards_per_sec).style(Style::default().fg(Color::Cyan)),
        parts[1],
    );
}

fn network(f: &mut Frame, area: Rect, s: &AppState) {
    let [gl, vk, cpu, unknown] = s.peers_by_backend();
    let known = gl + vk + cpu + unknown;
    let lines = vec![
        Line::from(vec!["peers known  ".dim(), Span::raw(known.to_string()).bold()]),
        Line::from(vec![
            "  OpenGL ".dim(),
            Span::raw(gl.to_string()),
            "  CPU ".dim(),
            Span::raw(cpu.to_string()),
            if vk > 0 { format!("  Vulkan {vk}").into() } else { "".into() },
        ]),
        Line::from(vec![
            "shards exchanged ".dim(),
            Span::raw((s.stats.served.shards + s.stats.used.shards).to_string()),
        ]),
        Line::from(vec![
            "jobs you ran ".dim(),
            Span::raw(s.stats.jobs.to_string()),
            "  retries ".dim(),
            Span::raw(s.stats.retries.to_string()),
        ]),
        Line::from(vec![
            "traffic ".dim(),
            Span::raw(mb(s.stats.served.bytes_in
                + s.stats.served.bytes_out
                + s.stats.used.bytes_in
                + s.stats.used.bytes_out)),
        ]),
    ];
    f.render_widget(Paragraph::new(lines).block(title("Network (seen from here)")), area);
}

fn neighbours(f: &mut Frame, area: Rect, s: &AppState) {
    let header = Row::new(["id", "address", "backend", "seen", "for them", "for you", "MB ↓/↑"])
        .style(Style::default().add_modifier(Modifier::BOLD));
    let rows = s.peers.iter().map(|p| {
        let bytes = p.for_them.bytes_in + p.for_us.bytes_in;
        let bytes_out = p.for_them.bytes_out + p.for_us.bytes_out;
        Row::new(vec![
            Cell::from(p.id.map(id).unwrap_or_else(|| "?".into())),
            Cell::from(p.addr.clone()),
            Cell::from(p.backend.map(|b| format!("{b:?}")).unwrap_or_else(|| "-".into())),
            Cell::from(p.last_seen.map(|d| format!("{} ago", dur(d))).unwrap_or_default()),
            Cell::from(format!("{}", p.for_them.shards)),
            Cell::from(format!("{}", p.for_us.shards)),
            Cell::from(format!("{}/{}", mb(bytes), mb(bytes_out))),
        ])
    });
    let table = Table::new(
        rows,
        [
            Constraint::Length(20),
            Constraint::Length(22),
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(9),
            Constraint::Length(8),
            Constraint::Min(14),
        ],
    )
    .header(header)
    .block(title(&format!("Neighbours ({}) · sorted by {}", s.peers.len(), s.sort.label())));
    f.render_widget(table, area);
}

fn logs(f: &mut Frame, area: Rect, s: &AppState) {
    let height = area.height.saturating_sub(2) as usize;
    let lines: Vec<Line> = s.logs.iter().skip(s.logs.len().saturating_sub(height)).map(|l| Line::raw(l.clone())).collect();
    f.render_widget(Paragraph::new(lines).block(title("Log")), area);
}

fn help(f: &mut Frame) {
    let area = f.area();
    let w = 56.min(area.width);
    let h = 11.min(area.height);
    let rect = Rect::new(area.x + (area.width - w) / 2, area.y + (area.height - h) / 2, w, h);
    let text = vec![
        Line::raw("q, Ctrl-C   quit"),
        Line::raw("Tab         change neighbour sort order"),
        Line::raw("?           toggle this help"),
        Line::raw(""),
        Line::raw("for them: shards you ran for that peer"),
        Line::raw("for you:  shards that peer ran for you"),
        Line::raw("? id:     traffic from an address the relay"),
        Line::raw("          never introduced (or shared IP)"),
    ];
    f.render_widget(Clear, rect);
    f.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }).block(title("Help")), rect);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::{NodeInfo, PeerRow, SortKey};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use struum_node::StatsSnapshot;
    use struum_types::network::NodeBackend;

    pub(crate) fn sample_state() -> AppState {
        let served = Tally { shards: 42, failed: 1, bytes_in: 50 << 20, bytes_out: 2 << 20, busy: Duration::from_secs(30) };
        let used = Tally { shards: 6, failed: 0, bytes_in: 1 << 20, bytes_out: 8 << 20, busy: Duration::from_secs(3) };
        AppState {
            node: NodeInfo {
                id: UID::new(0xabc123),
                ip: "192.168.1.20".parse().unwrap(),
                udp_port: 40001,
                tcp_port: 40002,
                backend: NodeBackend::OpenGL,
            },
            relay: "192.168.1.10:39001".parse().unwrap(),
            relay_status: RelayStatus::Connected,
            stats: StatsSnapshot {
                uptime: Duration::from_secs(120),
                served,
                used,
                jobs: 2,
                retries: 1,
                working: true,
                ..StatsSnapshot::default()
            },
            peers: vec![PeerRow {
                id: Some(UID::new(77)),
                addr: "192.168.1.30:34265".into(),
                backend: Some(NodeBackend::Cpu),
                last_seen: Some(Duration::from_secs(2)),
                for_them: served,
                for_us: used,
            }],
            shards_per_sec: vec![0, 1, 3, 2, 5],
            bytes_per_sec: vec![0, 1 << 20],
            logs: vec!["12:00:00 INFO  Relay introduced node 77".into()],
            sort: SortKey::ForThem,
            show_help: false,
        }
    }

    pub(crate) fn screen(state: &AppState, w: u16, h: u16) -> String {
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| render(f, state)).unwrap();
        let buf = term.backend().buffer().clone();
        (0..h)
            .map(|y| (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>().trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn renders_contribution_neighbours_and_logs() {
        let text = screen(&sample_state(), 110, 32);
        for needle in [
            "WORKING",
            "connected",
            "192.168.1.10:39001",
            "for others 42 shards  1 failed  in 50.0 MB",
            "0000000000abc123",
            "000000000000004d",
            "utilisation 25.0%",
            "peers known  1",
            "shards exchanged 48",
            "192.168.1.30:34265",
            "Relay introduced node 77",
            "sorted by shards for them",
        ] {
            assert!(text.contains(needle), "missing {needle:?} in:\n{text}");
        }
    }

    #[test]
    fn help_overlay_and_small_terminals() {
        let mut s = sample_state();
        s.show_help = true;
        assert!(screen(&s, 110, 32).contains("change neighbour sort order"));
        // Tiny terminals must not panic.
        let _ = screen(&s, 20, 5);
    }

    #[test]
    fn units() {
        assert_eq!(mb(512), "512 B");
        assert_eq!(mb(3 << 20), "3.0 MB");
        assert_eq!(dur(Duration::from_millis(950)), "950ms");
        assert_eq!(dur(Duration::from_secs(245)), "4m05s");
    }
}
