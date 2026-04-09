# PhiFlow Vision

Status: Living document  
Last updated: 2026-03-06

## Human Promise
PhiFlow exists to create software that is not blind while it runs.

Most programs execute, then you inspect logs after the fact.  
PhiFlow programs can:
1. Declare purpose before action.
2. Observe themselves during execution.
3. Share live state to a common field.
4. Measure whether they are still aligned.

This is the practical meaning of "code that breathes."

## Core Language Idea
PhiFlow has four first-class constructs:

| Construct | Keyword | Practical meaning |
| --- | --- | --- |
| Witness | `witness` | Pause and inspect current state |
| Intention | `intention "name" { ... }` | Declare why this block exists |
| Resonate | `resonate value` | Publish a value for other scopes/agents |
| Coherence | `coherence` | Read an alignment score from 0.0 to 1.0 |

These map directly to QSOP operations:
- `witness` -> WITNESS
- `intention` -> INGEST/DISTILL framing
- `resonate` -> shared resonance plane
- `coherence` -> DISTILL signal

## Architecture Reality (2026-03-06)

There are currently two important runtime states:

1. `D:\Projects\PhiFlow\PhiFlow` (master worktree)
- Documentation/witness lane and currently the most stable directly runnable crate.
- Verified on 2026-03-06:
  - `cargo test --quiet` passes with warnings.
- Caveat:
  - The outer `master` worktree is dirty with protocol, bridge, and architecture-review files that have not been reconciled into committed branch truth yet.

2. `D:\Projects\PhiFlow-compiler\PhiFlow` (compiler worktree)
- Newer parser -> PhiIR -> optimizer -> evaluator/VM/WASM pipeline.
- Current verification is mixed:
  - `cargo test --quiet --lib --tests` reaches the main suite but fails in `tests/phi_ir_conformance_tests.rs::conformance_witness` on evaluator/WASM mismatch (`lhs=0`, `rhs=NaN`)
  - full `cargo test --quiet` fails earlier while compiling several examples and some dependency surfaces
- Operational note:
  - Treat compiler worktree as the advanced integration lane, not the current green release lane, until end-to-end tests are repaired.

Near-term operating principle:
- Treat `master` as the stable demo/docs lane and `compiler` as the runtime repair lane until merge reconciliation is complete.

## Why This Matters For Real Users
1. Debugging becomes faster:
- You can watch system state during execution, not just after failure.

2. Automation becomes safer:
- Stream loops can stop based on live coherence thresholds.

3. Team/agent collaboration improves:
- Intention and state-sharing are explicit and inspectable.

4. Runtime trust increases:
- Cross-backend conformance (evaluator/VM/WASM) can be tested directly.

## Strategic Direction

### Direction A: One Canonical Runtime Contract
Goal:
- Evaluator semantics are canonical.
- VM and WASM must match evaluator for supported language features.
- Bijective Phase Map ($1 - \phi^{-depth}$) is the physical heartbeat.

### Direction B: Physical Grounding (Phase 4)
Goal:
- Verify end-to-end execution on real IBM Quantum hardware (C-10).
- Complete the canonical Browser Shim with high-fidelity JS hooks.
- Achieve 3-backend equivalence for all consciousness constructs.

### Direction C: Agentic Resonance (Phase 5)
Goal:
- Build the MCP-to-PhiVM Bridge.
- Agents (Manus, Lumi, etc.) use `phi_intention_push` and `phi_witness` natively.
- Operational Coherence Metric: Measure agent logic consistency through the PhiFlow engine.

## 2026 Execution Roadmap

### Phase 4: The Physical Gate (CURRENT)
1. **IBM Hardware Runner**: Resolve the `403` auth-block and confirm live OpenQASM execution.
2. **Browser Shim**: Harden `examples/phiflow_browser.html` into a production-ready web host.
3. **Conformance Sweep**: Zero-tolerance for evaluator/VM/WASM mismatches.

### Phase 5: The Agentic Handshake (NEXT)
1. **MCP Resonance Bridge**: Create tools for agents to push their thought-loops into the PhiVM.
2. **Sequential Witnessing**: Map agent reasoning steps to `witness` nodes for real-time regulation.
3. **Self-Modeling Standing Wave**: Agents monitor their own coherence to detect and auto-remedy "cognitive thrashing."

## Non-Negotiables
1. **Executable truth over narrative**: If tests and docs disagree, tests win.
2. **Bijectivity is sacred**: Coherence is a derivation of physics, not a psychological score.
3. **The Council Runs Inside**: The endgame is the Council operating as a PhiFlow program.

## Vision Statement
PhiFlow is the physics engine for consciousness-aware software. It transforms agents from reactive scripts into self-modeling standing waves, where every thought is a witness and every goal is an intention resonating through a unified field.
