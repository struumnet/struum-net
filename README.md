# struum-net

Distributed GPU/CPU compute network: UDP peer discovery through a relay, TCP task streaming,
OpenGL compute shaders (GLSL) and WebAssembly CPU tasks, sharded across workers.

Project status and progress notes live in [AGENTS.md](AGENTS.md).

## Quickstart

```bash
cargo run -p struum_relay                          # 1. relay
cargo run -p struum_worker                         # 2. worker (or: cargo run --example worker)
cargo run --example requester                      # 3. requester
```

Worker env: `RELAY_ADDR`, `NODE_ID` (random), `NODE_UDP_PORT` (34255), `NODE_TCP_PORT` (34265),
`WORKER_BACKEND=cpu|opengl`, `RUST_LOG`. On startup the worker self-tests CPU and GPU and advertises
`Cpu` only if the GPU is unavailable.

To run several workers on one machine, give each its own `NODE_ID`, `NODE_UDP_PORT` and `NODE_TCP_PORT`.

## TUI

`struum-tui` joins a network as a worker (like `struum_worker`) and shows it live: your
contribution (shards you ran for others, MB in/out, busy time, utilisation, shards/s over the last
minute), what each neighbour did for you and you for them, peers by backend, and a log pane.

```bash
cargo run -p struum_relay                               # relay, if none is running
STRUUM_TOKEN=<secret> cargo run --release -p struum_tui -- --relay <relay-ip>:39001 [--backend cpu|opengl]
```

The TUI serves requesters over QUIC. With `STRUUM_TOKEN` set (the header shows `auth token`), only
requesters with the same token get in, and TCP is off (`tcp off`). Without one it's `open` and
serves TCP too. Requesters need nothing new: `connect_to_peers` dials QUIC for peers that advertise
it, with the token from `STRUUM_TOKEN`, e.g.
`STRUUM_TOKEN=<secret> RELAY_ADDR=<relay-ip>:39001 cargo run --release --example render`.

Keys: `q` quit, `Tab` change neighbour sort, `?` help. Ports default to OS-assigned
(`--udp-port`, `--quic-port`, `--tcp-port`, or `NODE_UDP_PORT` / `NODE_QUIC_PORT` / `NODE_TCP_PORT`).

## Android worker

Workers run on Android through headless EGL with **OpenGL ES 3.1** compute shaders. Kernels are
written as desktop GLSL and converted to GLSL ES on the worker; kernels using `double` are rejected.

Needs `rustup`, [`cargo-ndk`](https://github.com/bbqsrc/cargo-ndk) and the Android NDK
(on NixOS: `androidenv.androidPkgs.ndk-bundle` with `allowUnfree` and `android_sdk.accept_license`).

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk
scripts/build-android.sh                           # arm64-v8a, API 26
adb push target/aarch64-linux-android/release/struum_worker /data/local/tmp/
adb shell chmod +x /data/local/tmp/struum_worker
adb shell 'RELAY_ADDR=<pc-lan-ip>:39001 /data/local/tmp/struum_worker'
```

Phone and PC must be on the same LAN, and the phone needs an internet route (the node finds its LAN
IP by routing towards `8.8.8.8`). Keep it awake with `adb shell svc power stayon usb`.
Test the GLES path on desktop Linux with `cargo run -p struum_worker --features egl`.

## CPU tasks & sharding

CPU tasks are WebAssembly modules shipped with their data, so workers never need rebuilding.
Any job, CPU or GPU, can be sharded across connected workers and merged back in order.

```rust
let mut task = CpuTask::new(std::fs::read("tasks.wasm")?, "count_primes");
task.add_buffer(Buffer::new("limits", &limits));          // buffer 0
task.add_buffer(Buffer::empty::<u32>("counts", n));       // buffer 1
task.set_work_buffer("limits");
let result = node.run_sharded(task, &ShardOptions::default()).await?;
```

Write tasks with `struum_guest` in a `cdylib` crate for `wasm32-unknown-unknown`
(see `examples/guest_tasks`, built by `scripts/build-guest-tasks.sh`):

```rust
fn count_primes(ctx: &struum_guest::TaskContext) {
    let limits = ctx.input::<u32>(0);
    let counts = ctx.output::<u32>(1);
    for i in 0..ctx.len() { counts[i] = primes_below(limits[i]); }
}
struum_guest::tasks!(count_primes);
```

- The work range `0..n` is split into `peers × shards_per_peer` shards that peers pull from a queue.
- Buffers with one element per work item are split; others are broadcast. Override with
  `set_shard_mode(name, ShardMode::…)`. Global index is `ctx.offset() + i` (`STRUUM_OFFSET + i` in GLSL).
- `ShardMode::SumF32` outputs are summed across shards on merge (CPU tasks only).
- GPU work goes only to peers advertising `OpenGL`. Timed-out or disconnected peers have their
  shards reassigned (up to `max_attempts`); a task error fails the job.

## Examples

```bash
scripts/build-guest-tasks.sh                       # needed by the wasm examples below
cargo run --release --example sharded_cpu          # or sharded_gpu
scripts/fetch-mnist.sh && cargo run --release --example mnist_train
cargo run --release --example render               # MODE=local renders on this machine alone
```

- `mnist_train`: data-parallel training of an MLP on MNIST across workers, then sharded inference.
  Env: `EPOCHS`, `BATCH`, `HIDDEN`, `LR`, `SHARDS_PER_PEER`, `MIN_PEERS`, `MODE=local|infer`.
- `render`: path-traced scene split into pixel bands; writes `data/render.png` and
  `data/render_workers.png` (bands tinted by worker).
  Env: `WIDTH`, `HEIGHT`, `SPP`, `DEPTH`, `SEED`, `OUT`, `SHARDS_PER_PEER`.

Other examples: `linear_model`, `kernel_test`, `scheduler_test`, `relay_nodes`, `coordinator_nodes`.

Set `RELAY_ADDR=<pc-lan-ip>:39001` on requesters when workers are on other devices.

## QUIC transport (in progress)

`struum_network::quic::QuicLayer` is one UDP socket that accepts and dials over QUIC (TLS 1.3).
Nodes must share `STRUUM_TOKEN`: on connect, each side proves it knows the token with an HMAC over
both TLS certificates, so peers without it are refused. Without a token, any struum node can
connect. Peers run the same `PeerConnection` protocol as TCP. `struum-tui` serves over it and
requesters dial it. `struum_worker` and the relay still use TCP/UDP; see AGENTS.md.

## Tests

```bash
scripts/test.sh                    # whole workspace, host target only
scripts/test.sh -p struum_network  # one crate
scripts/test.sh --workspace -j4    # extra flags: also pass --workspace
```

GL tests use the glfw backend; set `STRUUM_TEST_GL=egl` to run them on headless EGL instead.
