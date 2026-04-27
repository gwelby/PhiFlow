#!/bin/bash
# Post-merge setup for PhiFlow.
#
# Runs after a task is merged. Must be idempotent, non-interactive,
# and fail-fast. The full Nix environment is on PATH; stdin is /dev/null.
#
# We build the CLI binaries declared in Cargo.toml so that the next
# `target/debug/phic …` invocation in the examples README is ready to go
# without a cold compile during a user-facing run.

set -euo pipefail

cargo build --bins
