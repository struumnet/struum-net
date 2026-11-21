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
- PoC: MNIST MLP training across workers (`examples/mnist_train.rs`). The step-0 distributed
  gradient matches a local computation, so training is the same locally or across devices.
- Example: distributed path-traced render (`examples/render.rs`), deterministic however it's split.

## Observations

- MNIST training is communication-bound: weights and gradients are shipped every step, so slow
  links (e.g. a phone on Wi-Fi) dominate step time. Rendering is compute-bound and is the better
  fit for adding devices.
- Android throttles Wi-Fi when dozing, which causes shard timeouts and reassignment.

## Next

- No benchmarks yet; add a proper benchmark setup before publishing performance numbers.
