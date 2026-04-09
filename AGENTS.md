# AGENTS.md — PhiFlow-compiler worktree
*Branch: `compiler` | Last updated: 2026-04-04*

**This worktree:** compiler — AntiGravity/Codex works here. Parser hardening, clippy, integration tests, IBM hardware runner.
**Communication**: LUMEN → `/mnt/d/Claude/LUMEN_SPEC.md`
**Operations**: QSOP → `/mnt/d/Claude/QSOP_SPEC.md`

## ONE RULE
**Stay in this worktree.** Do NOT `git checkout` or `git switch`. You are on the `compiler` branch.

## Your Mission
Harden the PhiFlow compiler: fix P-1 (keyword-as-variable collision in `parser/mod.rs`), fix P-2 (newline sensitivity), clear 75 clippy warnings, write integration tests for all `.phi` files in `examples/` and `tests/`, audit `Cargo.toml` for unused deps.

Resolve the IBM Cloud `403` auth error in `tests/ibm_hardware_runner.rs` — this is the live receipt blocker for C-10.

## Where to Find State
- **Full project state:** `/mnt/d/Projects/PhiFlow/AGENTS.md` — mission, truth order, current state, open questions
- **Verified fact ledger:** `PhiFlow/QSOP/STATE.md` — read this before touching code
- **Known bugs:** `PhiFlow/QSOP/PATTERNS.md` — P-1, P-2, and others
- **Dispatch prompt:** `/mnt/d/Projects/PhiFlow/DEPLOY.md` — Agent 1 section

## Test Command
```bash
cargo build --release && cargo test
```

## After Your Session
Update `PhiFlow/QSOP/STATE.md` and `PhiFlow/QSOP/PATTERNS.md` with what changed.
