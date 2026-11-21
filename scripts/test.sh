#!/usr/bin/env bash
# Runs the workspace tests on the host target only (no Android / cross builds).
#
# GL tests use the headless EGL backend by default: the glfw backend creates a
# context per worker thread, which glfw doesn't support and which segfaults
# intermittently under the parallel test runner. Set STRUUM_TEST_GL=glfw to
# test the glfw backend anyway.
#
# Usage: scripts/test.sh                      whole workspace
#        scripts/test.sh -p struum_network    one crate
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
  set -- --workspace
fi

# The feature can only be named when struum_opengl itself is selected, so
# single-package runs (-p ...) build with the default backend.
features=()
if [ "${STRUUM_TEST_GL:-egl}" = egl ] && [[ " $* " == *" --workspace "* ]]; then
  features=(--features struum_opengl/egl)
fi
exec cargo test "${features[@]}" "$@"
