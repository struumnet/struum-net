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
- PoC: MNIST MLP training across workers (`examples/mnist_train.rs`). The step-0 distributed
  gradient matches a local computation, so training is the same locally or across devices.
- Example: distributed path-traced render (`examples/render.rs`), deterministic however it's split.

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

  Localhost hides bandwidth, so MB sent is the number that matters for Wi-Fi workers.

## Next

- Pipelining (2 shards in flight per peer) and speculative re-runs of the last shards.
