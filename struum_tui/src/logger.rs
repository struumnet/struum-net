//! A `log` backend that keeps the last lines in memory for the log pane.
//! Writing to stderr would corrupt the terminal UI.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use log::{Level, LevelFilter, Log, Metadata, Record};

/// Shared buffer of formatted log lines, oldest first.
#[derive(Debug, Clone, Default)]
pub struct LogBuffer {
    lines: Arc<Mutex<VecDeque<String>>>,
    capacity: usize,
}

impl LogBuffer {
    pub fn new(capacity: usize) -> Self {
        Self { lines: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))), capacity }
    }

    pub fn push(&self, line: String) {
        let mut lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        if lines.len() == self.capacity {
            lines.pop_front();
        }
        lines.push_back(line);
    }

    /// The last `n` lines, oldest first.
    pub fn tail(&self, n: usize) -> Vec<String> {
        let lines = self.lines.lock().unwrap_or_else(|e| e.into_inner());
        lines.iter().skip(lines.len().saturating_sub(n)).cloned().collect()
    }
}

struct RingLogger {
    buffer: LogBuffer,
    level: LevelFilter,
}

impl Log for RingLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Keep dependency chatter (wasmtime, cranelift, ...) out unless asked for.
        metadata.level() <= self.level
            && (metadata.target().starts_with("struum") || metadata.level() <= Level::Warn)
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
        self.buffer.push(format!("{h:02}:{m:02}:{s:02} {:<5} {}", record.level(), record.args()));
    }

    fn flush(&self) {}
}

static BUFFER: OnceLock<LogBuffer> = OnceLock::new();

/// Installs the ring logger (once per process) and returns its buffer.
/// The level comes from `RUST_LOG` if it is a plain level name, else `info`.
pub fn init(capacity: usize) -> LogBuffer {
    BUFFER
        .get_or_init(|| {
            let level = std::env::var("RUST_LOG")
                .ok()
                .and_then(|v| v.parse::<LevelFilter>().ok())
                .unwrap_or(LevelFilter::Info);
            let buffer = LogBuffer::new(capacity);
            let logger = RingLogger { buffer: buffer.clone(), level };
            if log::set_boxed_logger(Box::new(logger)).is_ok() {
                log::set_max_level(level);
            }
            buffer
        })
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_keeps_the_newest_lines() {
        let buf = LogBuffer::new(3);
        for i in 0..5 {
            buf.push(format!("line {i}"));
        }
        assert_eq!(buf.tail(10), vec!["line 2", "line 3", "line 4"]);
        assert_eq!(buf.tail(1), vec!["line 4"]);
    }

    #[test]
    fn logger_captures_struum_records() {
        let buf = init(100);
        log::info!(target: "struum_tui::test", "hello from the test");
        assert!(buf.tail(100).iter().any(|l| l.contains("hello from the test")));
    }
}
