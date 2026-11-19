//! Distributed path-traced render: the case where adding devices (even a phone) speeds things up.
//!
//! Each pixel costs thousands of ray/sphere tests, but a shard only carries a few config words
//! in and one u32 per pixel out, so computation dominates the network. Pixels are split into
//! bands that workers pull from a shared queue (faster devices take more bands). The result is
//! written as a PNG, plus a second PNG tinted by which worker rendered each band.
//!
//! Every pixel seeds its own RNG, so the image is the same no matter how it was split.
//!
//! 1. `scripts/build-guest-tasks.sh`
//! 2. `cargo run --release -p struum_relay`
//! 3. start workers (`cargo run --release -p struum_worker`, the Android worker, ...)
//! 4. `cargo run --release --example render`, then compare with `MODE=local`
//!
//! Env:
//! - `MODE`: `network` (default) or `local` (this machine's cores only, for comparison).
//! - `WIDTH` (800), `HEIGHT` (450), `SPP` samples per pixel (32), `DEPTH` max bounces (8),
//!   `SEED` scene layout (42), `OUT` (data/render.png).
//! - `SHARDS_PER_PEER` (8), `MIN_PEERS` (2), `RELAY_ADDR`, `STRUUM_WASM`,
//!   `NODE_UDP_PORT`, `NODE_TCP_PORT`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use struum_cpu::{CpuExecutor, CpuTask};
use struum_node::{Node, ShardOptions, ShardReport};
use struum_types::{Buffer, ShardMode, StruumError, network::*};

const DEFAULT_WASM: &str = "examples/guest_tasks/target/wasm32-unknown-unknown/release/struum_guest_tasks.wasm";

/// Tint colours for the worker map (RGB) and their names for the legend.
const PALETTE: &[([u8; 3], &str)] = &[
    ([230, 60, 60], "red"),
    ([60, 120, 230], "blue"),
    ([60, 200, 90], "green"),
    ([240, 190, 40], "yellow"),
    ([180, 80, 220], "purple"),
    ([40, 200, 210], "cyan"),
    ([250, 130, 40], "orange"),
    ([230, 90, 170], "pink"),
];

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn err(msg: impl Into<String>) -> StruumError {
    StruumError::NotFound(msg.into())
}

// ---------------------------------------------------------------- PNG

fn crc32(chunks: &[&[u8]]) -> u32 {
    let mut table = [0u32; 256];
    for (n, entry) in table.iter_mut().enumerate() {
        let mut c = n as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *entry = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for bytes in chunks {
        for &b in *bytes {
            crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Writes an RGBA image as a PNG using uncompressed (stored) deflate blocks: no dependencies.
fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) -> Result<(), StruumError> {
    let mut raw = Vec::with_capacity((width as usize * 4 + 1) * height as usize);
    for row in rgba.chunks_exact(width as usize * 4) {
        raw.push(0); // filter: none
        raw.extend_from_slice(row);
    }

    let mut zlib = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, block) in blocks.iter().enumerate() {
        zlib.push((i + 1 == blocks.len()) as u8);
        let len = block.len() as u16;
        zlib.extend(len.to_le_bytes());
        zlib.extend((!len).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    zlib.extend(((b << 16) | a).to_be_bytes());

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8], data: &[u8]| {
        png.extend((data.len() as u32).to_be_bytes());
        png.extend_from_slice(kind);
        png.extend_from_slice(data);
        png.extend(crc32(&[kind, data]).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend(width.to_be_bytes());
    ihdr.extend(height.to_be_bytes());
    ihdr.extend([8, 6, 0, 0, 0]); // 8-bit RGBA, no interlace
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &zlib);
    chunk(b"IEND", &[]);

    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(path, png).map_err(|e| err(format!("Cannot write {path}: {e}")))
}

// ---------------------------------------------------------------- render

fn render_task(wasm: Vec<u8>, width: u32, height: u32, spp: u32, depth: u32, seed: u32) -> Result<CpuTask, StruumError> {
    let mut task = CpuTask::new(wasm, "render");
    task.add_buffer(Buffer::new("config", &[width, height, spp, depth, seed]));
    task.add_buffer(Buffer::empty::<u32>("pixels", (width * height) as usize));
    task.set_work_buffer("pixels");
    task.set_shard_mode("config", ShardMode::Broadcast)?;
    Ok(task)
}

/// Blends each shard's band with its worker's colour and prints the legend.
fn worker_map(rgba: &[u8], report: &ShardReport) -> Vec<u8> {
    let mut workers: Vec<SocketAddr> = report.completed_by.keys().copied().collect();
    workers.sort();
    let colour: HashMap<SocketAddr, usize> = workers.iter().enumerate().map(|(i, a)| (*a, i % PALETTE.len())).collect();

    let total: usize = report.ranges.iter().map(|r| r.len()).sum();
    println!("\nWho rendered what:");
    for addr in &workers {
        let (rgb, name) = PALETTE[colour[addr]];
        let pixels: usize = report
            .ranges
            .iter()
            .zip(&report.completed_on)
            .filter(|(_, a)| *a == addr)
            .map(|(r, _)| r.len())
            .sum();
        println!(
            "  {name:<7} {addr}: {} band(s), {:.1}% of pixels  (rgb {rgb:?})",
            report.completed_by[addr],
            100.0 * pixels as f64 / total.max(1) as f64
        );
    }

    let mut out = rgba.to_vec();
    for (range, addr) in report.ranges.iter().zip(&report.completed_on) {
        let (rgb, _) = PALETTE[colour[addr]];
        for px in range.clone() {
            for c in 0..3 {
                let v = &mut out[px * 4 + c];
                *v = ((*v as u16 + rgb[c] as u16) / 2) as u8;
            }
        }
    }
    out
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mode = std::env::var("MODE").unwrap_or_else(|_| "network".to_string());
    let width: u32 = env_or("WIDTH", 800);
    let height: u32 = env_or("HEIGHT", 450);
    let spp: u32 = env_or("SPP", 32);
    let depth: u32 = env_or("DEPTH", 8);
    let seed: u32 = env_or("SEED", 42);
    let out_path = std::env::var("OUT").unwrap_or_else(|_| "data/render.png".to_string());
    let wasm_path = std::env::var("STRUUM_WASM").unwrap_or_else(|_| DEFAULT_WASM.to_string());
    let wasm = std::fs::read(&wasm_path)
        .map_err(|e| err(format!("Cannot read {wasm_path}: {e}. Run scripts/build-guest-tasks.sh first.")))?;
    if width < 2 || height < 2 {
        return Err(err("WIDTH and HEIGHT must be at least 2"));
    }

    let task = render_task(wasm, width, height, spp, depth, seed)?;
    let paths = width as f64 * height as f64 * spp as f64;
    println!("Rendering {width}x{height} at {spp} samples/pixel, up to {depth} bounces ({:.1}M camera rays)", paths / 1e6);

    let (result, report, elapsed): (CpuTask, Option<ShardReport>, Duration) = if mode == "local" {
        let exec = CpuExecutor::new()?;
        println!("Mode: local (this machine's cores only)");
        let started = Instant::now();
        let out = tokio::task::block_in_place(|| exec.execute(task))?;
        (out, None, started.elapsed())
    } else {
        let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
            .parse()
            .expect("Invalid RELAY_ADDR");
        let mut node = Node::<2048>::with_id(
            UID::new(6),
            env_or("NODE_UDP_PORT", 34260),
            env_or("NODE_TCP_PORT", 34270),
            NodeBackend::Cpu,
            1,
        )
        .await?;
        let peers = node.discover_peers(relay_addr, env_or("MIN_PEERS", 2), Duration::from_secs(2)).await?;
        let connected = node.connect_to_peers(&peers).await;
        if connected.is_empty() {
            return Err(err("no workers reachable"));
        }
        println!("Mode: network, {} worker(s): {:?}", connected.len(), connected);

        let opts = ShardOptions { shards_per_peer: env_or("SHARDS_PER_PEER", 8), ..ShardOptions::default() };
        let (work, report) = node.run_sharded_with_report(task, &opts).await?;
        if report.retries > 0 {
            println!("  {} band(s) reassigned; retired worker(s): {:?}", report.retries, report.retired);
        }
        let elapsed = report.elapsed;
        (work.into_cpu_task().ok_or_else(|| err("expected a CPU result"))?, Some(report), elapsed)
    };

    println!(
        "Rendered in {:.2?} ({:.2}M rays/s)",
        elapsed,
        paths / elapsed.as_secs_f64() / 1e6
    );

    let pixels = result.get_buffer("pixels").ok_or_else(|| err("no pixels"))?;
    write_png(&out_path, width, height, &pixels.data)?;
    println!("Wrote {out_path}");

    if let Some(report) = report {
        let map = worker_map(&pixels.data, &report);
        let map_path = out_path
            .strip_suffix(".png")
            .map(|base| format!("{base}_workers.png"))
            .unwrap_or_else(|| format!("{out_path}_workers.png"));
        write_png(&map_path, width, height, &map)?;
        println!("Wrote {map_path} (each band tinted by the worker that rendered it)");
        println!("\nCompare with this machine alone: MODE=local cargo run --release --example render");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_known_value() {
        // CRC-32 of "IEND" is the well-known AE426082.
        assert_eq!(crc32(&[b"IEND"]), 0xAE42_6082);
    }
}
