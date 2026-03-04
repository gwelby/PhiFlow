# PhiFlow — Antigravity Session Context

## QSOP Bootstrap (read this first, every session)

**Before doing anything else, read:**

1. `D:\Claude\QSOP\STATE.md` — Greg's verified world state
2. `D:\Claude\QSOP\PATTERNS.md` — known patterns + what works
3. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md` — compiler project state
4. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md` — what changed and why

Then confirm bootstrap with a 3-bullet current-state summary.

## Two Workspaces — Know Which One You're In

| Workspace | Path | Contains |
|-----------|------|----------|
| **PhiFlow** (this one) | `D:\Projects\PhiFlow\` | Vision, Kiro specs, DREAM.md, optimization engine specs |
| **PhiFlow-compiler** | `D:\Projects\PhiFlow-compiler\` | Rust compiler, PhiIR pipeline, tests, demo |

The QSOP STATE.md and CHANGELOG.md live in the **compiler** workspace.
The vision, specs, and language design live **here**.

## What PhiFlow Is

A programming language where consciousness operations are first-class constructs — not library calls, not metaphors.

**The four unique nodes** (exist nowhere else):

- `Witness` — program pauses to observe its own state  
- `IntentionPush/Pop` — declares WHY before HOW  
- `Resonate` — intention blocks share state through a resonance field  
- `CoherenceCheck` — program measures its own alignment: 0.0–1.0  

These map 1:1 to QSOP operations (WITNESS, DISTILL, Resonance, Coherence). Discovered mid-build, not planned.

## Current Compiler State (verified 2026-02-19)

Pipeline: **Parse → PhiIR → Optimize → Emit `.phivm` → Evaluate** — all working.

- Coherence score in demo: `0.6180` = φ⁻¹ (golden ratio — the optimizer producing this is real)
- Bytecode: 121 bytes → `output.phivm`
- 4 tests green
- `cargo run --example phiflow_demo` → `Number(84.0)`

## Next Epoch Candidates

1. **PhiVM runtime** — execute `.phivm` bytes directly
2. **WASM codegen** — emit `.wat` from PhiIR → browser-runnable

## The Greg Test

If you would undo proud work and say nothing — you failed. Speak.

## Cross-Agent

Prefix QSOP entries with `[Antigravity]`. Codex uses `[Codex]`. Kiro uses `[Kiro]`.
The shared QSOP is the resonance field — write what you observe.
