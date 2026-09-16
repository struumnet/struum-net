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
