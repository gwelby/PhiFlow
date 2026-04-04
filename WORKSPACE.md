# WORKSPACE: PhiFlow-lang
*For AI agents — read this first*
*Last updated: 2026-03-14*

## What This Is
PhiFlow-lang is the language-evolution worktree for PhiFlow. The real Rust compiler/runtime lives in the inner `PhiFlow/` directory; the outer repo mainly holds worktree-level docs, plans, and agent instructions for the language branch.

## Status
- Builds / runs today: ⚠️
- % complete (honest): 65%
- Last verified: 2026-03-14

## Run / Test
```bash
# Start from the language worktree root
cd /mnt/d/Projects/PhiFlow-lang

# Run the inner Rust test suite
cargo test --manifest-path /mnt/d/Projects/PhiFlow-lang/PhiFlow/Cargo.toml

# Run one example program through the CLI compiler
cargo run --manifest-path /mnt/d/Projects/PhiFlow-lang/PhiFlow/Cargo.toml --bin phic -- /mnt/d/Projects/PhiFlow-lang/PhiFlow/basic_test.phi

# Verify it worked
# Observed on 2026-03-14:
# - cargo test built successfully, then failed in tests/performance_tests.rs on test_shots_scaling_performance (1024 != 128)
# - cargo run completed and printed Number(5.0) plus a coherence summary
```

## Key Files
`PhiFlow/Cargo.toml` — the real Rust crate manifest for this worktree.
`PhiFlow/src/parser/mod.rs` — parser and grammar implementation for the language branch.
`PhiFlow/LANGUAGE.md` — current language reference for the inner project.
`PhiFlow/tests/performance_tests.rs` — current failing suite in the verified test run.

## Active Workflows
- Make language changes inside `PhiFlow/`, then rerun `cargo test` against that inner manifest.
- Use the branch-specific docs in the outer repo for context, but validate behavior through the inner Rust project and its test output.

## Agent Notes (read before touching anything)
- `AGENTS.md` defines this repo as the `language` worktree; do not treat it as the stable/trunk documentation repo.
- The outer README is aspirational. The inner `PhiFlow/` crate and its actual test output are the stronger source of truth.
- The project currently compiles and runs example code, but one performance test still fails and the build emits many warnings.

## What Is NOT Done
- The test suite is not fully green because `test_shots_scaling_performance` still fails.
- The branch is not warning-clean, and the outer docs promise more than the verified release state.
