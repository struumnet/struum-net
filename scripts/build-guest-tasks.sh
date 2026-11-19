#!/usr/bin/env bash
# Builds the example CPU tasks (examples/guest_tasks) to WebAssembly.
set -euo pipefail
cd "$(dirname "$0")/../examples/guest_tasks"
rustup target add wasm32-unknown-unknown >/dev/null
cargo build --release --target wasm32-unknown-unknown "$@"
OUT="$(pwd)/target/wasm32-unknown-unknown/release/struum_guest_tasks.wasm"
echo "Built: $OUT"
