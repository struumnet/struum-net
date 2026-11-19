#!/usr/bin/env bash
# Cross-compiles the struum_worker binary for Android (arm64-v8a).
#
# Requirements:
#   - rustup        (for the aarch64-linux-android std)
#   - cargo-ndk     (cargo install cargo-ndk)
#   - Android NDK   (ANDROID_NDK_HOME pointing at it)
#
# Usage: scripts/build-android.sh [extra cargo args]
# Env:   ANDROID_API (default 26), ANDROID_ABI (default arm64-v8a)
set -euo pipefail

cd "$(dirname "$0")/.."

ANDROID_API="${ANDROID_API:-26}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"

case "$ANDROID_ABI" in
  arm64-v8a)   RUST_TARGET=aarch64-linux-android ;;
  armeabi-v7a) RUST_TARGET=armv7-linux-androideabi ;;
  x86_64)      RUST_TARGET=x86_64-linux-android ;;
  *) echo "Unsupported ANDROID_ABI: $ANDROID_ABI" >&2; exit 1 ;;
esac

if [[ -z "${ANDROID_NDK_HOME:-}" ]]; then
  echo "ANDROID_NDK_HOME is not set; point it at your Android NDK." >&2
  exit 1
fi
command -v cargo-ndk >/dev/null || { echo "cargo-ndk not found (cargo install cargo-ndk)." >&2; exit 1; }

rustup target add "$RUST_TARGET"
cargo ndk -t "$ANDROID_ABI" -P "$ANDROID_API" build -p struum_worker --release "$@"

OUT="target/$RUST_TARGET/release/struum_worker"
echo
echo "Built: $OUT"
echo "Deploy: adb push $OUT /data/local/tmp/ && adb shell chmod +x /data/local/tmp/struum_worker"
