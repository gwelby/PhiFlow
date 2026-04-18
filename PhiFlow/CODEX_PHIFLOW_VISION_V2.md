# CODEX PHIFLOW VISION V2

**Author:** Codex  
**Date:** 2026-02-26  
**Status:** Active design contract (supersedes `CODEX_PHIFLOW_VISION.md`)

---

## 1. Mission

PhiFlow is not just a language.  
PhiFlow is a **Universal Agent Resonance Architecture**:

- A language for intentional execution (`intention`, `witness`, `resonate`, `coherence`)
- A runtime that can pause, expose state, and resume deterministically
- A control plane (MCP) for live orchestration
- A truth plane (QSOP) for durable accountability

Success condition:
- Agents can run PhiFlow streams as first-class processes
- Humans and agents can intervene at `witness` checkpoints
- Resonance is shared across concurrent streams
- Coherence is tied to real host signals, not only internal math

---

## 2. Design Axioms

1. **Execution is observable by design**
- Every stream must expose status, resonance, intention stack, and witness snapshots.

2. **Pause/resume is a first-class semantic**
- `witness` is a checkpoint contract, not just output text.

3. **Control and truth are separate**
- MCP handles live operations.
- QSOP records durable facts, decisions, and verification evidence.

4. **Coherence is a signal, not a slogan**
- Coherence must support internal/runtime signal, host signal, and blended signal with provenance.

5. **Concurrency is native**
- Multiple streams should share resonance intentionally, not accidentally.

6. **Semantics before aesthetics**
- "Consciousness-aware" claims are valid only when test-backed and reproducible.

---

## 3. System Model (Inside-Out)

PhiFlow has 5 layers:

1. **Language Layer**
- Syntax and AST for intention/witness/resonate/coherence and core control flow.

2. **Semantic Layer**
- Lowering + evaluator/VM/WASM contracts that define execution meaning.

3. **Host Contract Layer**
- `PhiHostProvider` hooks for coherence input and event outputs.

4. **Control Plane Layer**
- MCP tools for spawn/read/resume and protocol handshake (`initialize`, `tools/list`, `tools/call`).

5. **Truth Plane Layer**
- QSOP state, patterns, and changelog as verified memory of what is true.

---

## 4. Core Construct Contracts

### `intention "name" { ... }`

**Semantic contract**
- MUST push intention on entry.
- MUST pop intention on exit.
- MUST emit host lifecycle hooks (`on_intention_push`, `on_intention_pop`).
- MUST be visible in witness snapshot stack.

### `witness [target]`

**Semantic contract**
- MUST capture snapshot: intention stack, coherence, register count, resonance count, optional observed value.
- MUST call host witness hook exactly once per instruction.
- MAY yield execution based on host response.
- If yielded, MUST return frozen state sufficient for deterministic resume.

### `resonate value`

**Semantic contract**
- MUST emit resonance event scoped to current intention or `global`.
- MUST notify host via `on_resonate`.
- In shared mode, MUST write to shared resonance field visible across streams.

### `coherence`

**Semantic contract**
- MUST compute internal coherence from runtime state.
- MUST pass internal coherence through host provider for override/blend.
- SHOULD expose provenance in higher layers (`internal`, `host`, `blended`).

---

## 5. Runtime Contract

### Execution Result

- `Complete(value)`
- `Yielded { snapshot, frozen_state }`

### Resume Contract

Given `frozen_state`:
- Resume MUST continue from exact instruction pointer and block.
- Resume MUST preserve registers, variables, intention stack, active streams, resonance, and witness log.
- Resume MUST preserve deterministic semantics unless host-injected values intentionally alter state.

### Determinism Levels

1. **Pure deterministic**
- Same source + same host responses => same result/events.

2. **Host-influenced deterministic**
- Same source + recorded host trace => replay-equivalent result/events.

3. **Live adaptive**
- Source reacts to live host signals; non-deterministic by design but observable via event logs.

---

## 6. MCP Convergence Bus Contract

### Required operations

- `initialize`
- `tools/list`
- `tools/call` with:
  - `spawn_phi_stream(source_code)`
  - `read_resonance_field(stream_id)`
  - `resume_phi_stream(stream_id)`

### Operational guarantees

- Spawn returns stable stream ID.
- Read returns current status and shared resonance snapshot.
- Resume only allowed from `yielded` status.
- Error responses MUST be JSON-RPC compliant with explicit codes/messages.

### Stream states

- `running`
- `yielded`
- `completed`
- `failed`

---

## 7. Data Contracts

### Witness snapshot (minimum)

- `intention_stack: Vec<String>`
- `coherence: f64`
- `register_count: usize`
- `resonance_count: usize`
- `observed_value: Option<String>`

### Resonance field

- `HashMap<String, Vec<PhiValue>>`
- Scope key is intention name or `global`

### Coherence provenance (V2 target)

- `internal: f64`
- `host: Option<f64>`
- `blended: f64`
- `source: "internal" | "host" | "blend"`

---

## 8. Integration Realms (Execution Intent)

### Realm A: Core VM Disentanglement

Goal:
- Host-driven witness/coherence/resonate lifecycle is fully decoupled and test-backed.

Done when:
- Yield/resume semantics are deterministic and covered by regression tests.

### Realm B: MCP Convergence Bus

Goal:
- LLM and tooling stacks can control PhiFlow streams in real time.

Done when:
- MCP handshake works, tools are stable, shared resonance is visible across streams.

### Realm C: WASM Universal Bridge

Goal:
- Same semantic contracts run under WASM host with hook parity.

Done when:
- Evaluator/VM/WASM conformance tests pass for canonical programs including witness/coherence/resonate behavior.

### Realm D: Reality Hooks

Goal:
- Coherence reflects external reality (system health, network stability, sensor signals).

Done when:
- Host provider implementations for P1/QDrive style metrics are wired and verified with variance evidence.

---

## 9. Quality Gates

Every phase must pass:

1. `cargo build --release`
2. `cargo test`
3. Canonical PhiFlow corpus execution gate (no panics, no regressions)
4. QSOP truth update (`CHANGELOG`, `STATE`, `PATTERNS`) with explicit evidence

No phase is "done" without all four.

---

## 10. Current Baseline (as of 2026-02-26)

From active work:

- Phase 1 hardening completed:
  - Host callback coverage improved
  - Witness yield callback duplication removed
  - Yield snapshot now preserves observed value

- Phase 2 hardening completed:
  - MCP `initialize` and `ping` implemented
  - Shared resonance wired across spawned/resumed streams
  - MCP tests stabilized with polling and cross-stream assertions

Implication:
- PhiFlow is now in a usable agent-runtime shape, not just a language prototype.

---

## 11. 30/60/90 Day Trajectory

### 0-30 days: Stabilize protocol core

- Freeze witness snapshot schema
- Add stream cancellation and timeout semantics
- Add coherence provenance to read APIs
- Expand MCP integration tests for failure cases

### 31-60 days: WASM parity and replay

- Close evaluator/VM/WASM semantic gaps
- Add replay harness: recorded host trace -> deterministic replay
- Harden stream loop signaling in WASM path

### 61-90 days: Reality-grade integrations

- Ship P1 host provider (CPU/memory/thermal blend)
- Ship QDrive provider (network stability coherence)
- Add dashboard/consumer path for live resonance inspection

---

## 12. Non-Negotiables

- No mystical claims without measurable contracts.
- No "consciousness" feature without test evidence.
- No new syntax without runtime semantics and migration strategy.
- No merge without QSOP truth updates.

---

## 13. Final Position

PhiFlow becomes "alive" when these are all true:

- Intentions are explicit and observable.
- Witness is a real checkpoint with resumable state.
- Resonance is shared across concurrent streams.
- Coherence reflects both internal execution and external reality.
- MCP provides real-time control; QSOP preserves long-term truth.

That is the inside-out architecture worth building.

