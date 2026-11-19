//! PoC: train a neural network on MNIST across the struum network, then run it.
//!
//! Data-parallel SGD: every step the minibatch is split across all workers (PC, phone, ...).
//! Each worker computes the gradient of its slice in WebAssembly (`mlp_grad` in
//! examples/guest_tasks), the slices' gradients are summed (a `SumF32` buffer, i.e. an
//! all-reduce), and the requester applies Adam. Afterwards the trained model classifies the
//! 10k test images, again sharded across the workers (`mlp_predict`).
//!
//! 1. `scripts/fetch-mnist.sh` and `scripts/build-guest-tasks.sh`
//! 2. `cargo run --release -p struum_relay`
//! 3. start workers (`cargo run --release -p struum_worker`, the Android worker, ...)
//! 4. `cargo run --release --example mnist_train`
//!
//! Env:
//! - `MODE`: `train` (default), `infer` (load saved weights and only run inference), or `local`
//!   (train in-process on all local cores with no network, as a baseline).
//! - `EPOCHS` (3), `BATCH` (1000), `HIDDEN` (128), `LR` (0.002), `SHARDS_PER_PEER` (2),
//!   `MIN_PEERS` (2), `RELAY_ADDR`, `STRUUM_WASM`, `MNIST_DIR` (data/mnist),
//!   `WEIGHTS` (data/mnist_mlp.bin), `NODE_UDP_PORT`, `NODE_TCP_PORT`.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use struum_cpu::{CpuExecutor, CpuTask};
use struum_node::{Node, ShardOptions, ShardReport};
use struum_types::{Buffer, ShardMode, StruumError, network::*};

const DEFAULT_WASM: &str = "examples/guest_tasks/target/wasm32-unknown-unknown/release/struum_guest_tasks.wasm";
const PIXELS: usize = 28 * 28;
const CLASSES: usize = 10;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn err(msg: impl Into<String>) -> StruumError {
    StruumError::NotFound(msg.into())
}

// ---------------------------------------------------------------- data

struct Dataset {
    images: Vec<u8>,
    labels: Vec<u32>,
}

impl Dataset {
    fn len(&self) -> usize {
        self.labels.len()
    }

    fn image(&self, i: usize) -> &[u8] {
        &self.images[i * PIXELS..(i + 1) * PIXELS]
    }

    fn load(dir: &str, prefix: &str) -> Result<Self, StruumError> {
        let read = |name: String| {
            let path = format!("{dir}/{name}");
            std::fs::read(&path).map_err(|e| err(format!("Cannot read {path}: {e}. Run scripts/fetch-mnist.sh first.")))
        };
        let be = |b: &[u8], at: usize| u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize;

        let images = read(format!("{prefix}-images-idx3-ubyte"))?;
        let labels = read(format!("{prefix}-labels-idx1-ubyte"))?;
        if be(&images, 0) != 0x803 || be(&labels, 0) != 0x801 || be(&images, 8) * be(&images, 12) != PIXELS {
            return Err(err(format!("{dir}: not MNIST IDX files")));
        }
        let n = be(&images, 4);
        Ok(Self {
            images: images[16..16 + n * PIXELS].to_vec(),
            labels: labels[8..8 + n].iter().map(|&l| l as u32).collect(),
        })
    }
}

/// Tiny deterministic RNG (xorshift64*), so no extra dependency is needed.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }
    fn normal(&mut self) -> f32 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next_u64() % (i as u64 + 1)) as usize);
        }
    }
}

// ---------------------------------------------------------------- model

/// Same layout as `mlp::Dims` in the guest: `[W1 | b1 | W2 | b2]`.
struct Model {
    hidden: usize,
    params: Vec<f32>,
}

impl Model {
    fn param_count(hidden: usize) -> usize {
        hidden * PIXELS + hidden + CLASSES * hidden + CLASSES
    }

    /// He initialisation, zero biases.
    fn new(hidden: usize, rng: &mut Rng) -> Self {
        let mut params = Vec::with_capacity(Self::param_count(hidden));
        let s1 = (2.0 / PIXELS as f32).sqrt();
        let s2 = (2.0 / hidden as f32).sqrt();
        params.extend((0..hidden * PIXELS).map(|_| rng.normal() * s1));
        params.extend(std::iter::repeat_n(0.0, hidden));
        params.extend((0..CLASSES * hidden).map(|_| rng.normal() * s2));
        params.extend(std::iter::repeat_n(0.0, CLASSES));
        Self { hidden, params }
    }

    fn config(&self) -> [u32; 3] {
        [PIXELS as u32, self.hidden as u32, CLASSES as u32]
    }

    fn save(&self, path: &str) -> Result<(), StruumError> {
        let mut bytes = b"STRUUMMLP1".to_vec();
        bytes.extend((self.hidden as u32).to_le_bytes());
        bytes.extend(self.params.iter().flat_map(|p| p.to_le_bytes()));
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, bytes).map_err(|e| err(format!("Cannot write {path}: {e}")))
    }

    fn load(path: &str) -> Result<Self, StruumError> {
        let bytes = std::fs::read(path).map_err(|e| err(format!("Cannot read {path}: {e}. Train first.")))?;
        if bytes.len() < 14 || &bytes[..10] != b"STRUUMMLP1" {
            return Err(err(format!("{path} is not a struum MLP file")));
        }
        let hidden = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
        let params: Vec<f32> = bytes[14..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if params.len() != Self::param_count(hidden) {
            return Err(err(format!("{path} has the wrong number of parameters")));
        }
        Ok(Self { hidden, params })
    }
}

struct Adam {
    lr: f32,
    t: i32,
    m: Vec<f32>,
    v: Vec<f32>,
}

impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Self { lr, t: 0, m: vec![0.0; n], v: vec![0.0; n] }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        self.t += 1;
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        for i in 0..params.len() {
            let g = grads[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            params[i] -= self.lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + eps);
        }
    }
}

// ---------------------------------------------------------------- tasks

fn grad_task(wasm: &[u8], model: &Model, images: &[u8], labels: &[u32]) -> Result<CpuTask, StruumError> {
    let mut task = CpuTask::new(wasm.to_vec(), "mlp_grad");
    task.add_buffer(Buffer::from_rows("images", images, PIXELS)?);
    task.add_buffer(Buffer::new("labels", labels));
    task.add_buffer(Buffer::new("config", &model.config()));
    task.add_buffer(Buffer::new("params", &model.params));
    task.add_buffer(Buffer::empty::<f32>("grads", model.params.len() + 2));
    task.set_work_buffer("labels");
    task.set_shard_mode("config", ShardMode::Broadcast)?;
    task.set_shard_mode("params", ShardMode::Broadcast)?;
    task.set_shard_mode("grads", ShardMode::SumF32)?;
    Ok(task)
}

fn predict_task(wasm: &[u8], model: &Model, images: &[u8]) -> Result<CpuTask, StruumError> {
    let n = images.len() / PIXELS;
    let mut task = CpuTask::new(wasm.to_vec(), "mlp_predict");
    task.add_buffer(Buffer::from_rows("images", images, PIXELS)?);
    task.add_buffer(Buffer::new("config", &model.config()));
    task.add_buffer(Buffer::new("params", &model.params));
    task.add_buffer(Buffer::empty::<u32>("preds", n));
    task.set_work_buffer("preds");
    task.set_shard_mode("config", ShardMode::Broadcast)?;
    task.set_shard_mode("params", ShardMode::Broadcast)?;
    Ok(task)
}

/// Where tasks run: sharded across network workers, or on this machine only.
enum Runner {
    Network { node: Box<Node<2048>>, opts: ShardOptions, shards: HashMap<SocketAddr, usize> },
    Local(CpuExecutor),
}

impl Runner {
    async fn run(&mut self, task: CpuTask) -> Result<CpuTask, StruumError> {
        match self {
            Runner::Network { node, opts, shards } => {
                let (work, report): (_, ShardReport) = node.run_sharded_with_report(task, opts).await?;
                for (addr, n) in report.completed_by {
                    *shards.entry(addr).or_default() += n;
                }
                if report.retries > 0 {
                    println!(
                        "    ! {} shard(s) reassigned, retired peer(s): {:?}",
                        report.retries, report.retired
                    );
                }
                work.into_cpu_task().ok_or_else(|| err("expected a CPU result"))
            }
            Runner::Local(exec) => tokio::task::block_in_place(|| exec.execute(task)),
        }
    }

    /// Shards completed per worker since the last call, as "addr: n, ...".
    fn take_split(&mut self) -> String {
        match self {
            Runner::Network { shards, .. } => {
                let mut v: Vec<_> = shards.drain().collect();
                v.sort();
                v.iter().map(|(a, n)| format!("{a}: {n}")).collect::<Vec<_>>().join(", ")
            }
            Runner::Local(_) => "local".to_string(),
        }
    }
}

async fn evaluate(runner: &mut Runner, wasm: &[u8], model: &Model, test: &Dataset) -> Result<(Vec<u32>, f32, Duration), StruumError> {
    let started = Instant::now();
    let out = runner.run(predict_task(wasm, model, &test.images)?).await?;
    let elapsed = started.elapsed();
    let preds = out.get_buffer("preds").ok_or_else(|| err("no preds"))?.as_slice::<u32>().to_vec();
    let correct = preds.iter().zip(&test.labels).filter(|(p, l)| p == l).count();
    Ok((preds, correct as f32 / test.len() as f32, elapsed))
}

/// Prints test digits side by side as ASCII art with their predicted labels.
fn show_digits(test: &Dataset, preds: &[u32], picks: &[usize]) {
    const SHADES: &[u8] = b" .:-=+*#%@";
    for row in (0..28).step_by(2) {
        let line: Vec<String> = picks
            .iter()
            .map(|&i| {
                let img = test.image(i);
                (0..28)
                    .map(|col| {
                        let v = (img[row * 28 + col] as usize + img[(row + 1) * 28 + col] as usize) / 2;
                        SHADES[v * (SHADES.len() - 1) / 255] as char
                    })
                    .collect()
            })
            .collect();
        println!("  {}", line.join("  "));
    }
    let captions: Vec<String> = picks
        .iter()
        .map(|&i| {
            let mark = if preds[i] == test.labels[i] { "ok" } else { "WRONG" };
            format!("{:<28}", format!("predicted {} (is {}) {mark}", preds[i], test.labels[i]))
        })
        .collect();
    println!("  {}", captions.join("  "));
}

#[tokio::main]
async fn main() -> Result<(), StruumError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let mode = std::env::var("MODE").unwrap_or_else(|_| "train".to_string());
    let epochs: usize = env_or("EPOCHS", 3);
    let batch: usize = env_or("BATCH", 1000);
    let hidden: usize = env_or("HIDDEN", 128);
    let lr: f32 = env_or("LR", 0.002);
    let weights_path = std::env::var("WEIGHTS").unwrap_or_else(|_| "data/mnist_mlp.bin".to_string());
    let mnist_dir = std::env::var("MNIST_DIR").unwrap_or_else(|_| "data/mnist".to_string());
    let wasm_path = std::env::var("STRUUM_WASM").unwrap_or_else(|_| DEFAULT_WASM.to_string());
    let wasm = std::fs::read(&wasm_path)
        .map_err(|e| err(format!("Cannot read {wasm_path}: {e}. Run scripts/build-guest-tasks.sh first.")))?;

    let train = Dataset::load(&mnist_dir, "train")?;
    let test = Dataset::load(&mnist_dir, "t10k")?;
    println!("MNIST: {} training / {} test images", train.len(), test.len());

    let mut runner = if mode == "local" {
        let exec = CpuExecutor::new()?;
        println!("Running locally on this machine only (baseline, no network)");
        Runner::Local(exec)
    } else {
        let relay_addr: SocketAddr = std::env::var("RELAY_ADDR")
            .unwrap_or_else(|_| "127.0.0.1:39001".to_string())
            .parse()
            .expect("Invalid RELAY_ADDR");
        let mut node = Node::<2048>::with_id(
            UID::new(5),
            env_or("NODE_UDP_PORT", 34259),
            env_or("NODE_TCP_PORT", 34269),
            NodeBackend::Cpu,
            1,
        )
        .await?;
        let peers = node.discover_peers(relay_addr, env_or("MIN_PEERS", 2), Duration::from_secs(2)).await?;
        let connected = node.connect_to_peers(&peers).await;
        if connected.is_empty() {
            return Err(err("no workers reachable"));
        }
        println!("Connected to {} worker(s): {:?}", connected.len(), connected);
        let opts = ShardOptions { shards_per_peer: env_or("SHARDS_PER_PEER", 2), ..ShardOptions::default() };
        Runner::Network { node: Box::new(node), opts, shards: HashMap::new() }
    };

    let model = if mode == "infer" {
        let model = Model::load(&weights_path)?;
        println!("Loaded {} parameters from {weights_path}", model.params.len());
        model
    } else {
        train_model(&mut runner, &wasm, &train, &test, epochs, batch, hidden, lr).await?
    };

    // --- Run the model: classify the test set across the network ---
    println!("\nInference on {} test images...", test.len());
    let (preds, accuracy, elapsed) = evaluate(&mut runner, &wasm, &model, &test).await?;
    println!(
        "  accuracy {:.2}% in {:?} ({:.0} images/s) [{}]",
        accuracy * 100.0,
        elapsed,
        test.len() as f64 / elapsed.as_secs_f64(),
        runner.take_split()
    );
    let mut picks: Vec<usize> = (0..test.len()).filter(|&i| preds[i] == test.labels[i]).take(3).collect();
    picks.extend((0..test.len()).filter(|&i| preds[i] != test.labels[i]).take(2));
    println!();
    show_digits(&test, &preds, &picks);

    if mode != "infer" {
        model.save(&weights_path)?;
        println!("\nSaved weights to {weights_path} (MODE=infer reuses them)");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn train_model(
    runner: &mut Runner,
    wasm: &[u8],
    train: &Dataset,
    test: &Dataset,
    epochs: usize,
    batch: usize,
    hidden: usize,
    lr: f32,
) -> Result<Model, StruumError> {
    let mut rng = Rng(0x5EED_1234_ABCD_0001);
    let mut model = Model::new(hidden, &mut rng);
    let mut adam = Adam::new(model.params.len(), lr);
    let p = model.params.len();
    let steps = train.len().div_ceil(batch);
    println!(
        "Model 784-{hidden}-10 ({p} parameters), {epochs} epoch(s) x {steps} steps of {batch} images, Adam lr {lr}"
    );

    let mut order: Vec<usize> = (0..train.len()).collect();
    let started = Instant::now();
    for epoch in 1..=epochs {
        rng.shuffle(&mut order);
        let epoch_start = Instant::now();
        let (mut loss_sum, mut correct_sum, mut seen) = (0.0f64, 0.0f64, 0usize);

        for (step, idx) in order.chunks(batch).enumerate() {
            let mut images = Vec::with_capacity(idx.len() * PIXELS);
            for &i in idx {
                images.extend_from_slice(train.image(i));
            }
            let labels: Vec<u32> = idx.iter().map(|&i| train.labels[i]).collect();

            let task = grad_task(wasm, &model, &images, &labels)?;
            let check = (epoch == 1 && step == 0 && matches!(runner, Runner::Network { .. })).then(|| task.clone());
            let step_start = Instant::now();
            let out = runner.run(task).await?;
            let step_time = step_start.elapsed();
            let sums = out.get_buffer("grads").ok_or_else(|| err("no grads"))?.as_slice::<f32>().to_vec();

            if let Some(task) = check {
                // The network's summed gradient must match a single local computation.
                let local = tokio::task::block_in_place(|| CpuExecutor::new()?.execute(task))?;
                let want = local.get_buffer("grads").ok_or_else(|| err("no grads"))?.as_slice::<f32>();
                let n = idx.len() as f32;
                let max_diff = sums[..p]
                    .iter()
                    .zip(&want[..p])
                    .map(|(a, b)| ((a - b) / n).abs())
                    .fold(0.0f32, f32::max);
                println!("  step-0 gradient check vs local: max diff {max_diff:.2e}");
                if max_diff > 1e-4 {
                    return Err(StruumError::TaskError(format!("gradient mismatch {max_diff}")));
                }
            }

            let n = idx.len() as f32;
            let grads: Vec<f32> = sums[..p].iter().map(|g| g / n).collect();
            adam.step(&mut model.params, &grads);
            loss_sum += sums[p] as f64;
            correct_sum += sums[p + 1] as f64;
            seen += idx.len();

            if step % 10 == 0 || step + 1 == steps {
                println!(
                    "  epoch {epoch} step {:>2}/{steps}: loss {:.4}, train acc {:5.2}%, {:>4} ms/step [{}]",
                    step + 1,
                    loss_sum / seen as f64,
                    100.0 * correct_sum / seen as f64,
                    step_time.as_millis(),
                    runner.take_split()
                );
            }
        }

        let (_, accuracy, _) = evaluate(runner, wasm, &model, test).await?;
        runner.take_split();
        println!(
            "epoch {epoch}: {:?}, test accuracy {:.2}%",
            epoch_start.elapsed(),
            accuracy * 100.0
        );
    }
    println!("Training took {:?}", started.elapsed());
    Ok(model)
}
