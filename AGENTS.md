# AGENTS.md

Progress and status notes for struum-net. Build, run and contributing docs live in
[README.md](README.md); keep them there and keep this file to what has been done and what's next.

## Done

- Relay-based UDP peer discovery with long-polling introductions; TCP task streaming.
- OpenGL compute kernels (GLSL) on desktop; headless EGL / OpenGL ES 3.1 backend for Android workers.
- Standalone `struum_worker` binary with CPU/GPU self-test and backend selection.
- WebAssembly CPU tasks (`struum_guest`), split across all cores on each worker.
- Sharded jobs (`run_sharded`) for CPU and GPU: pull-based shard queue, split/broadcast buffers,
  `SumF32` all-reduce outputs, shard reassignment on timeout/disconnect.
  `ShardReport` lists each shard's range (`ranges`) and the worker that completed it (`completed_on`).
- Blob cache (`WORKREF` / `CACHEMISS`): wasm modules and buffers of 4 KiB or more go to each peer
  once per connection, then by xxh3 hash. Workers keep an LRU store per requester
  (`Node::blob_cache_bytes`, default `STRUUM_BLOB_CACHE_MB` = 1024); an evicted blob gets a
  `CACHEMISS` reply and the shard is resent in full, without retiring the peer.
  `ShardOptions::blob_cache` turns it off.
- Pipelining: each peer keeps `ShardOptions::in_flight` shards outstanding (default 2). Connections
  have separate read and write halves, and replies are checked against the shard that's due.
  If a peer fails, every shard it held is requeued. A read or write cut off part-way marks the
  connection broken, so it's never misread later.
- Speculation (`ShardOptions::speculate`, on by default): once the queue is empty, an idle peer
  re-runs the oldest outstanding shard after it has taken more than twice the mean round trip,
  at most once per shard. The first result wins and the job returns without waiting. The slow
  peer's task drains the late reply in the background and holds that connection's reader until
  it does. `ShardReport` adds `speculative_runs` and `speculative_wins`.
- PoC: MNIST MLP training across workers (`examples/mnist_train.rs`). The step-0 distributed
  gradient matches a local computation, so training is the same locally or across devices.
- Example: distributed path-traced render (`examples/render.rs`), deterministic however it's split.

- Concurrent serving: `Node::server()` gives a cloneable `PeerServer`, and `struum_worker` and
  `struum-tui` serve each requester in its own task. `JobScheduler::run` waits on a per-job
  reply, so nothing has to drain the completion channel.
- Peers joining mid-job: `Node::run_sharded_with_joins` takes a channel of new
  `PeerConnection`s (`PeerConnection::connect`). Each starts pulling from the job's queue.

- QUIC transport (`struum_network::quic`): `QuicLayer` accepts and dials on one UDP socket with
  per-node self-signed certificates. Token auth: a `Hello` carries
  `HMAC-SHA256(STRUUM_TOKEN, id ‖ own cert fp ‖ peer cert fp)` each way. Handshakes run in
  their own tasks. Each peer's packets go over one bidirectional stream inside a
  `PeerConnection` (now transport-agnostic), so dispatch and `PeerServer` work over QUIC
  unchanged (`struum_node/tests/quic_sharding.rs`).
- TUI over QUIC: `NodeDetails` / `IntroductionPacket` carry `quic_port` (a wire change, so all
  nodes need the same build). `struum-tui` serves QUIC (`struum_tui::serving`), and with
  `STRUUM_TOKEN` set it turns TCP off. The header shows `quic`, `tcp off` and `auth token|open`.
  `Node::connect_to_peers` dials QUIC peers (endpoint started on first use, token from env) and
  never falls back to TCP for them. Checked live: relay + TUI + `bench`; the right token runs the
  job, a wrong one is rejected.

## Observations

- MNIST training is communication-bound: weights and gradients are shipped every step, so slow
  links (e.g. a phone on Wi-Fi) dominate step time. Rendering is compute-bound and is the better
  fit for adding devices.
- Android throttles Wi-Fi when dozing, which causes shard timeouts and reassignment.

- Host-only test setup: `scripts/test.sh` (`STRUUM_TEST_GL=egl` for headless GL), with wire-format, TCP/UDP
  transport and relay tests.
- Benchmark: `examples/bench.rs` reports wall time and MB on the wire; `ShardReport` carries
  `bytes_sent`, `bytes_received`, `bytes_saved`, `round_trip` and `cache_misses`.
- glfw backend no longer crashes when worker threads exit (glfw is initialised once and never
  terminated).
- Live node stats (`Node::stats()`): shards, bytes and busy time served for / used from each peer.
- LAN speed work, measured with `cargo run --release --example bench` (4 in-process workers,
  `broadcast` = 5 steps with an 8 MB buffer sent to every shard):

  | Change | broadcast wall | broadcast MB sent |
  |---|---|---|
  | Baseline | 4028 ms | 339.7 |
  | serde_bytes, zero-copy sends, 4 MiB socket buffers | 4031 ms | 339.7 |
  | No per-thread copies of broadcast buffers in the wasm executor | 1861 ms | 339.7 |
  | Blob cache (`BLOB_CACHE=0` to compare: 1770 ms, 339.7 MB) | 1565 ms | 34.7 |
  | Pipelining + speculation (`IN_FLIGHT=1 SPECULATE=0`: ~1540 ms) | ~1570 ms | 34.8 |

  Localhost has no link latency to hide, so pipelining is neutral there. Speculating at 1× the
  mean round trip cost ~10% (copies compete for the same CPU), so the threshold is 2×.
  Both are for real devices: Wi-Fi round trips and a slow phone holding the last shard.

  Localhost hides bandwidth, so MB sent is the number that matters for Wi-Fi workers.

## Next

- Measure pipelining and speculation on real devices (desktop plus a phone on Wi-Fi) with
  `RELAY_ADDR=... cargo run --release --example bench`.
- WAN plan, phase 2 (on `feat/wan-quic-transport`):
  - Relay rewrite over QUIC: registry with expiry, `ListPeers`, `Connect`, `PunchRequest`.
  - Hole punching (LAN candidates first), then relay fallback (`Forward` streams).
  - `node.join(relay)` / `node.peers()`, wired into `struum_worker`, `struum-tui` and the examples.
  - `struum_worker` still serves TCP only: give it the same QUIC serving as the TUI.
  - Then delete the TCP path.
