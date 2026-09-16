# struum-net

Distributed GPU/CPU compute network with UDP peer discovery, TCP task streaming, and OpenGL compute shaders.

## Quickstart: Running the Distributed Network

Open 3 terminals to run a relay server and two peer nodes:

### 1. Relay Server
```bash
cargo run -p struum_relay
```

### 2. Worker Node (Listener)
```bash
cargo run -p struum_node --bin listener
```

### 3. Client Node (Hello)
```bash
cargo run -p struum_node --bin hello
```

The nodes register with the relay via UDP, receive peer introductions, establish a direct TCP connection, serialize the GLSL compute task & buffer bindings, execute on GPU worker threads, and return the computed results.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `RELAY_UDP_PORT` | Relay UDP bind port | `39001` |
| `RELAY_TCP_PORT` | Relay TCP bind port | `39002` |
| `RELAY_ADDR` | Relay server address | `127.0.0.1:39001` |
| `NODE_UDP_PORT` | Node UDP port | `34255` (listener) / `34256` (hello) |
| `NODE_TCP_PORT` | Node TCP port | `34265` (listener) / `34266` (hello) |
| `NODE_ID` | Node identifier | `1` (listener) / `2` (hello) |

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
