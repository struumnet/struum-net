# struum-net

Distributed GPU/CPU compute network with UDP peer discovery, TCP task streaming, and OpenGL compute shaders.

## Quickstart: Running the Distributed Network

Open 3 terminals to run a relay server and two peer nodes:

### 1. Relay Server
```bash
RELAY_UDP_PORT=39001 RELAY_TCP_PORT=39002 cargo run -p struum_relay
```

### 2. Worker Node (Listener)
```bash
NODE_ID=1 NODE_UDP_PORT=34255 NODE_TCP_PORT=34265 RELAY_ADDR=127.0.0.1:39001 cargo run -p struum_node --bin listener
```

### 3. Client Node (Hello)
```bash
NODE_ID=2 NODE_UDP_PORT=34256 NODE_TCP_PORT=34266 RELAY_ADDR=127.0.0.1:39001 cargo run -p struum_node --bin hello
```

The nodes register with the relay via UDP, receive peer introductions, establish a direct TCP connection, serialize the Kernel (GLSL source and buffer bindings), execute on GPU worker threads, and return the computed results.

## Examples

Run any example from `examples/`:

```bash
# Relay node discovery & peer introduction
cargo run --example relay_nodes

# Coordinator peer map & task dispatching
cargo run --example coordinator_nodes

# Local OpenGL compute shader (linear regression)
cargo run --example linear_model

# GLSL shader generation & struct mangling
cargo run --example kernel_test

# Multi-threaded worker scheduler
cargo run --example scheduler_test
```

## Tests

```bash
cargo test --workspace
```
