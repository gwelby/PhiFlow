# PhiFlow Vision

Status: Living document  
Last updated: 2026-02-26

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

## Architecture Reality (2026-02-26)

There are currently two important runtime states:

1. `D:\Projects\PhiFlow\PhiFlow` (master worktree)
- Historical/original crate and documentation lane.
- Local `cargo test` currently fails in `tests/performance_tests.rs` because simulator shots scaling is inconsistent.

2. `D:\Projects\PhiFlow-compiler\PhiFlow` (compiler worktree)
- Newer parser -> PhiIR -> optimizer -> evaluator/VM/WASM pipeline.
- Verified passing:
  - `cargo build --release`
  - `cargo test --quiet`
  - corpus sweep test (`test_all_phi_files_parse_and_execute`)

Near-term operating principle:
- Treat compiler worktree as runtime truth until merge reconciliation is complete.

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

Outcome:
- Fewer semantic regressions and clearer release criteria.

### Direction B: Human-First Developer Experience
Goal:
- Documentation written first for operators and builders, then for compiler internals.
- Every major feature documented with a plain-language example and expected runtime behavior.

Outcome:
- Easier onboarding and less interpretation drift.

### Direction C: Integration-Driven Value
Goal:
- Connect PhiFlow to existing assets in `D:\Projects` to produce immediate utility.

Outcome:
- Faster validation through practical use cases, not isolated demos.

## D:\Projects Integration Roadmap

The following candidates are grounded in existing local projects.

| Candidate | Path | Integration idea | Priority |
| --- | --- | --- | --- |
| UniversalProcessor | `D:\Projects\UniversalProcessor` | Add PhiFlow execution adapter (`process(kind=\"phiflow\", payload=...)`) to run `.phi` workloads as a processor kind | High |
| ResonanceMatrix | `D:\Projects\ResonanceMatrix` | Feed live `witness`/`resonate` stream events to dashboard for cross-agent observability | High |
| MCP | `D:\Projects\MCP` | Expose PhiFlow compile/run/diagnostics as MCP tools (`parse_phi`, `run_phi`, `watch_stream`) | High |
| P1_Companion | `D:\Projects\P1_Companion` | Use mobile sensor vectors as optional coherence provider input channel | Medium |
| QDrive | `D:\Projects\QDrive` | Store/retrieve resonance snapshots and execution traces as portable artifacts | Medium |
| Quantum-Fonts | `D:\Projects\Quantum-Fonts` | Optional visual identity layer for witness/resonance dashboards and docs | Low |

## 90-Day Execution Plan

### Phase 1 (Stabilize Runtime Contract)
1. Reconcile `master` and `compiler` runtime paths.
2. Fix simulator shots scaling bug in master.
3. Enforce release gates:
- `cargo build --release`
- `cargo test --quiet`
- `.phi` corpus sweep gate

### Phase 2 (Operational Integrations)
1. Build UniversalProcessor adapter for `.phi` execution.
2. Publish MCP tools for compile/run/diagnostics.
3. Connect stream JSON output to ResonanceMatrix live panel.

### Phase 3 (Sensor and Field Extensions)
1. Add optional P1_Companion sensor ingestion path.
2. Add QDrive artifact pathway for resonance/evidence transport.
3. Add operator-focused playbooks for production use.

## Non-Negotiables
1. Executable truth over narrative:
- If tests and docs disagree, tests win.

2. Small reversible changes:
- Prefer narrow commits and clear verification steps.

3. Cross-worktree discipline:
- Work in the correct lane; merge with explicit evidence.

4. Human readability:
- Every major behavior must have a plain-language explanation and example.

## Vision Statement
PhiFlow should become a practical language for adaptive, observable, purpose-aware software where humans and agents can collaborate in real time with shared trust in runtime behavior.
