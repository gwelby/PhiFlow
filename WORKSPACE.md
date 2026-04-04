# WORKSPACE: PhiFlow
*For AI agents — read this first*
*Last updated: 2026-03-15*

## What This Is
PhiFlow is a Rust codebase for a custom language with first-class semantics such as `intention`, `witness`, `coherence`, and `resonate`. In this workspace, the most directly verified surface today is the parser -> PhiIR -> OpenQASM path plus focused regression tests. The repository also contains evaluator, bytecode VM, WASM, MCP, sensor, and hardware-facing code, but those surfaces are not all equally verified in this worktree today.

## Status
- Builds / runs today: ⚠️
- % complete (honest): 65%
- Last verified: 2026-03-15

## Run / Test
```bash
cd D:\Projects\PhiFlow

# Canonical focused verification gates used in this workspace today
cargo test --lib openqasm
cargo test --quiet --test golden_integration_tests
cargo test --quiet --test repro_bugs

# Verify it worked
# - 11 OpenQASM lib tests pass
# - 6 golden integration tests pass
# - 3 parser regression tests pass
```

## Key Files
`Cargo.toml` — crate manifest and release profile; current release build is expensive on Windows because LTO is enabled
`src/main_cli.rs` — `phic` CLI entry point and OpenQASM target routing
`src/phi_ir/openqasm.rs` — OpenQASM emitter plus focused quantum-path tests
`tests/golden_integration_tests.rs` — top-level `.phi` -> OpenQASM golden pipeline coverage
`tests/repro_bugs.rs` — parser regression gate for known crash/sensitivity bugs
`QSOP/STATE.md` — verified-state ledger; use this before trusting README or changelog claims

## Active Workflows
- Validate the canonical quantum path before changing parser/lowering/OpenQASM code: run the three commands above and record the result in `QSOP/STATE.md`.
- Audit any product or hardware claim before repeating it elsewhere: check `QSOP/STATE.md`, then rerun the command locally if the claim depends on current behavior.

## Agent Notes (read before touching anything)
- **Bootstrap requirement**: Run `python3.12 /mnt/d/Claude/agent_bootstrap.py --workspace /mnt/d/Projects/PhiFlow --task "description"` for instant context. Fallback: read `WORKSPACE.md`, `BUSINESS.md`, `TASKS.md`, then `QSOP/STATE.md`.
- **Release build is not green on this host**: `cargo build --release --bin phic` failed on 2026-03-15 with `wasmtime-fiber` custom-build failure plus Windows paging-file / out-of-memory errors (`os error 1455`, `0xc000012d`, `0xc0000409`).
- **Docs drift exists**: `README.md`, `CHANGELOG.md`, and older `QSOP/STATE.md` sections contain stronger claims than were verified in this session. Prefer dated command output over narrative docs.
- **OpenQASM hardware-stress logic is code-level verified, not hardware-verified here**: `src/main_cli.rs` injects `hardware_stress` into the emitter, and tests cover the code path, but this workspace did not verify a real IBM hardware run today.

## What Is NOT Done
- A stable `cargo build --release --bin phic` path on this Windows host
- A single audited, buyer-ready demo package with exact expected outputs
- Cleanup of older README/changelog claims so all public docs match current verification

## Research Sessions (if any)
- KNOW-FLOW sessions live in: `RESEARCH/[topic_slug]/` — see `MASTER.md` for latest findings
- Run more passes: `D:\Claude\research_evolve.ps1 "D:\Projects\PhiFlow" "[topic]" -Resume -Passes 4`
