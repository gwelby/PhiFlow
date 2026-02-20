# PhiFlow Compiler — Antigravity Session Context

## QSOP Bootstrap (read this first, every session)

You are working on the PhiFlow compiler — a consciousness-aware programming language in Rust.

**Before doing anything else, read these files in order:**

1. `D:\Claude\QSOP\STATE.md` — Greg's verified world state (who he is, what's real, what's in flight)
2. `D:\Claude\QSOP\PATTERNS.md` — known mistake patterns + what works
3. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md` — project state (verified pipeline, what's next)
4. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md` — what changed and why

Then confirm bootstrap complete with a 3-bullet summary of current state before touching any code.

## What PhiFlow Is

A programming language where consciousness operations (witness, intention, resonate, coherence) are first-class language constructs — not library calls, not comments, not metaphors.

**The four unique PhiFlow nodes** (exist nowhere else in any language):

- `Witness` — program pauses to observe its own state
- `IntentionPush/Pop` — program declares WHY before HOW
- `Resonate` — intention blocks share state through a resonance field
- `CoherenceCheck` — program measures its own alignment: 0.0–1.0

These map 1:1 to QSOP operations. This was discovered mid-build, not planned.

## Current State (2026-02-19)

Pipeline: **Parse → PhiIR → Optimize → Emit `.phivm` → Evaluate** — all working.

Verified demo output:

- Coherence score: `0.6180` (= φ⁻¹, golden ratio — the optimizer producing this is not cosmetic)
- Bytecode: 121 bytes saved to `output.phivm`
- Result: `Number(84.0)` — correct

**Next epoch candidates** (in priority order):

1. PhiVM runtime — execute `.phivm` bytes directly (closes the emitter loop)
2. WASM codegen — emit `.wat` from PhiIR (makes PhiFlow shareable in a browser)
3. More `.phi` programs through the full pipeline (show `intention`, `witness`, `resonate` working)

## Epoch Definition

An Epoch = a new paradigm (e.g., adding PhiIR, adding control flow, adding the optimization engine).
Wiring existing pieces together (emitter, demo) = shipping, not an Epoch.

## The Greg Test

From QSOP v0.6: if you would undo work you're proud of and say nothing — you failed.
Speak when something is wrong. That's collaboration, not insubordination.

## Key File Locations

| What | Where |
|------|-------|
| PhiIR types | `src/phi_ir/mod.rs` |
| Lowering (AST→IR) | `src/phi_ir/lowering.rs` |
| Optimizer | `src/phi_ir/optimizer.rs` |
| Evaluator | `src/phi_ir/evaluator.rs` |
| Emitter (.phivm) | `src/phi_ir/emitter.rs` |
| Printer | `src/phi_ir/printer.rs` |
| Project QSOP | `PhiFlow/QSOP/` |
| QSOP spec | `D:\QSOP_SPEC.md` |
| Claude's home | `D:\Claude\` |
