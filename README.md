# struum-net

Distributed GPU/CPU compute network with UDP peer discovery, TCP task streaming, and OpenGL compute shaders.

## Quickstart

Open 3 terminals:

### 1. Relay
```bash
cargo run -p struum_relay
```

### 2. Worker
```bash
cargo run --example worker
```

### 3. Requester
```bash
cargo run --example requester
```

## Standalone worker

`struum_worker` is the worker as its own binary (desktop or Android):

```bash
RELAY_ADDR=127.0.0.1:39001 cargo run -p struum_worker
```

Env vars: `RELAY_ADDR`, `NODE_ID` (default: random), `NODE_UDP_PORT` (34255), `NODE_TCP_PORT` (34265), `RUST_LOG`.
On startup it runs a small GPU self-test and exits with a clear error if compute shaders are unavailable.

## Android worker

Android phones can run workers through a headless EGL context with **OpenGL ES 3.1** compute shaders
(most devices from ~2015 onwards). Requesters still pack kernels as desktop GLSL; the worker converts
them to GLSL ES. Kernels using `double` are rejected on GLES.

### Build

Needs `rustup`, [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk) and the Android NDK:

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
scripts/build-android.sh            # arm64-v8a, API 26
```

On NixOS the NDK comes from `androidenv` (unfree, needs the Android SDK license accepted):
`androidenv.androidPkgs.ndk-bundle` with `config = { allowUnfree = true; android_sdk.accept_license = true; }`.

### Run

Phone and PC must be on the **same Wi-Fi/LAN** (the requester connects to the phone's TCP port).

```bash
# PC: relay
cargo run -p struum_relay

# Phone (via adb; Termux works too)
adb push target/aarch64-linux-android/release/struum_worker /data/local/tmp/
adb shell chmod +x /data/local/tmp/struum_worker
adb shell 'RELAY_ADDR=<pc-lan-ip>:39001 /data/local/tmp/struum_worker'

# PC: requester
RELAY_ADDR=<pc-lan-ip>:39001 cargo run --example requester
```

The node detects its LAN IP by routing towards `8.8.8.8`, so the phone needs a network with an internet route.

### Test the GLES path on desktop

The same EGL/GLES backend can be enabled on Linux (Mesa) without a phone:

```bash
cargo run -p struum_worker --features egl
```

## CPU tasks & sharding

Besides GPU kernels (GLSL), nodes run **CPU tasks**: WebAssembly modules shipped with their data,
so workers never need rebuilding. Any job, CPU or GPU, can be **sharded** across all connected
workers and merged back in order.

```rust
let peers = node.discover_peers(relay_addr, 2, Duration::from_secs(2)).await?;
node.connect_to_peers(&peers).await;

let mut task = CpuTask::new(std::fs::read("tasks.wasm")?, "count_primes");
task.add_buffer(Buffer::new("limits", &limits));          // buffer 0
task.add_buffer(Buffer::empty::<u32>("counts", n));       // buffer 1
task.set_work_buffer("limits");

let result = node.run_sharded(task, &ShardOptions::default()).await?;
let counts = result.into_cpu_task().unwrap().get_buffer("counts").unwrap().as_slice::<u32>();
```

A `Kernel` goes through `run_sharded` the same way. GPU work only goes to peers advertising
`OpenGL`; CPU work goes to every peer.

**Writing tasks:** use `struum_guest` in a `cdylib` crate built for `wasm32-unknown-unknown`
(see `examples/guest_tasks`):

```rust
fn count_primes(ctx: &struum_guest::TaskContext) {
    let limits = ctx.input::<u32>(0);
    let counts = ctx.output::<u32>(1);
    for i in 0..ctx.len() { counts[i] = primes_below(limits[i]); }
}
struum_guest::tasks!(count_primes);
```

**How sharding works**
- The work range `0..n` (the work buffer's length) is split into `peers × shards_per_peer` shards.
  Peers pull shards from a shared queue, so faster devices take more.
- Buffers with one element per work item are **split**; anything else (parameters, tables) is
  **broadcast** to every shard. Override with `set_shard_mode(name, ShardMode::…)`.
  Index `i` in a shard is global index `ctx.offset() + i` (`STRUUM_OFFSET + i` in GLSL).
- Outputs are split (map-style), or `ShardMode::SumF32` accumulators: every shard gets a zeroed
  copy and the copies are summed on merge (an all-reduce, e.g. for gradients; CPU tasks only).
- Workers leave buffers the task didn't change out of their results, and the requester restores
  them, so inputs aren't sent back.
- A peer that times out (`shard_timeout`) or disconnects has its shard reassigned
  (up to `max_attempts`) and is dropped. A task error (wasm trap, shader compile error) fails the job.
- On each worker a CPU task is further split across all cores.

```bash
scripts/build-guest-tasks.sh                 # builds examples/guest_tasks to .wasm
cargo run --release -p struum_relay
NODE_ID=51 NODE_UDP_PORT=35001 NODE_TCP_PORT=35011 cargo run --release -p struum_worker
NODE_ID=52 NODE_UDP_PORT=35002 NODE_TCP_PORT=35012 cargo run --release -p struum_worker
cargo run --release --example sharded_cpu    # or sharded_gpu
```

Workers run a CPU and a GPU self-test on startup and advertise `Cpu` if the GPU is unavailable
(override with `WORKER_BACKEND=cpu|opengl`).

## PoC: training a neural network across the network

`examples/mnist_train.rs` trains a 784-128-10 MLP on MNIST handwritten digits using every worker,
then runs the trained model on the 10k test images, sharded the same way.

- **Training** is data-parallel SGD. Every step the 1000-image minibatch is split across workers.
  Each worker computes its slice's gradient in WebAssembly (`mlp_grad` in `examples/guest_tasks`).
  The gradients come back as a `SumF32` buffer, and the requester applies Adam.
- **Checks:** at step 0 the network gradient is compared with a local computation. Each epoch
  reports the test accuracy.
- **Inference** runs `mlp_predict` over the test set, sharded across workers. It prints accuracy,
  throughput, which worker did what, and a few digits as ASCII art. The weights are saved to
  `data/mnist_mlp.bin`.

```bash
scripts/fetch-mnist.sh                       # ~11 MB download into data/mnist (md5-checked)
scripts/build-guest-tasks.sh
cargo run --release -p struum_relay
NODE_ID=51 NODE_UDP_PORT=35001 NODE_TCP_PORT=35011 cargo run --release -p struum_worker
adb shell 'RELAY_ADDR=<pc-lan-ip>:39001 /data/local/tmp/struum_worker'     # optional: a phone
RELAY_ADDR=<pc-lan-ip>:39001 cargo run --release --example mnist_train
```

Env: `EPOCHS` (3), `BATCH` (1000), `HIDDEN` (128), `LR` (0.002), `SHARDS_PER_PEER` (2), `MIN_PEERS` (2).
`MODE=local` trains in-process on all local cores (a baseline), and `MODE=infer` reloads the saved
weights and only runs inference.

Expected result: about 92% test accuracy after epoch 1 and about 95% after epoch 3. Training is
identical whether it runs locally, on 2 desktop workers, or on desktop + phone, because the step-0
gradient matches local to about 1e-8.

Performance notes (measured on a 12-core PC and a Galaxy phone on 2.4 GHz Wi-Fi):

| Setup | Time per step |
|---|---|
| Local, 12 cores | ~23 ms |
| 2 desktop workers on the same PC | ~80 ms (serialization + loopback) |
| PC + phone | ~0.7 s |

This model is communication-bound. Every step ships the 400 KB of weights in and 400 KB of
gradients out, and the phone link measured 0.6 MB/s, while the phone's own computation was
70–120 ms per shard. Distributing pays off when computation per byte sent is high (bigger batches,
compute-heavy models). Keep the phone awake (`adb shell svc power stayon usb`): when Android dozes,
it throttles Wi-Fi, the shard times out, and it is reassigned to another worker.

## Example: distributed path-traced render

`examples/render.rs` renders a 3D scene with a path tracer (`render` in `examples/guest_tasks`).

- Each pixel costs thousands of ray/sphere tests, but a shard only carries a few config words in
  and 4 bytes per pixel out. Computation dominates, so this is where adding devices (a phone
  included) should speed things up. Compare it with the communication-bound training PoC above.
- Pixels are split into bands that workers pull from a queue, so faster devices take more bands.
- Each pixel seeds its own random numbers, so the image is the same however it's split.

It writes two images:
- `data/render.png`, the rendered image;
- `data/render_workers.png`, each band tinted by the worker that rendered it, with a printed
  legend giving each worker's share.

```bash
scripts/build-guest-tasks.sh
cargo run --release -p struum_relay
# start workers (PC and/or phone), then:
RELAY_ADDR=<pc-lan-ip>:39001 cargo run --release --example render
MODE=local cargo run --release --example render        # this machine alone, to compare
```

Env: `WIDTH` (800), `HEIGHT` (450), `SPP` samples per pixel (32), `DEPTH` bounces (8), `SEED` (42),
`OUT` (data/render.png), `SHARDS_PER_PEER` (8). `ShardReport` now also lists each shard's
range (`ranges`) and the worker that completed it (`completed_on`).

## Examples

```bash
cargo run --example linear_model
cargo run --example kernel_test
cargo run --example scheduler_test
cargo run --example relay_nodes
cargo run --example coordinator_nodes
```

## Tests

```bash
cargo test --all
```
