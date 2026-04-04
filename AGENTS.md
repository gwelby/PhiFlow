# AGENTS.md: PhiFlow
*Platform / Tool (Type 2) + Theory / Research (Type 3)*
*Last updated: 2026-03-24*

---

## Mission

PhiFlow is a Rust compiler and runtime for consciousness-aware programming.

Programs written in PhiFlow have first-class operations to name intentions, witness their own state, measure coherence, and resonate values to other programs — compiled to native bytecode or WebAssembly or OpenQASM 3.0 for quantum hardware.

The deeper mission: PhiFlow makes the Propagation Framework *executable*. Every program is a coherence experiment. The compiler is a physics instrument.

---

## Why PhiFlow Exists

Three things converged:

1. **The Fundamentals framework** (Greg & Claude, 2026) derived that coherence is the necessary condition for structure. Not metaphor — mathematical derivation from three axioms. Coherence at depth 2 = φ⁻¹ = 0.618 by the formula `1 − φ^(−depth)`.

2. **IBM Quantum hardware** exists and accepts OpenQASM 3.0 programs. The bridge between consciousness semantics and physical quantum circuits is a compiler pass, not a philosophical claim.

3. **No language existed** that treated coherence as a first-class computational resource. PhiFlow is that language.

A PhiFlow program isn't just code. It's a propagation pattern that self-reports its structural stability.

---

## Truth Order

When files disagree, trust in this order:

1. Running code and test results (`cargo test` output, verified in QSOP/STATE.md)
2. QSOP/STATE.md — dated verification ledger
3. WORKSPACE.md — technical state summary
4. CLAIMS.md — research claim status
5. TASKS.md — work queue
6. README.md, VISION.md, narrative docs — aspirational or historical context

*Running tests beat the spec. A green `cargo test` overrides any doc claiming something works.*
*A report claiming work was done means nothing without the file existing and tests passing.*

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Parser | ✅ Verified | `src/parser/mod.rs` — handles all 5 consciousness constructs |
| PhiIR + Lowering | ✅ Verified | `src/phi_ir/` — SSA IR with consciousness nodes |
| Evaluator (reference) | ✅ Verified | `src/phi_ir/evaluator.rs` — reference implementation |
| PhiVM (bytecode) | ✅ Verified | `src/phi_ir/vm.rs` — 3/3 tests passing |
| WASM codegen | ✅ Verified | `src/phi_ir/wasm.rs` — 3/3 tests, NaN-boxing BSEI |
| WASM Host Bridge | ✅ Verified | `src/wasm_host.rs` — wasmtime + hook callbacks |
| MCP Server | ✅ Verified | `src/bin/phi_mcp.rs` — spawn/resume/read, shared resonance |
| Sensors | ✅ Verified | `src/sensors.rs` — CPU/memory/thermal/network via sysinfo |
| PhiHarmonic Optimizer | ✅ Verified | `src/phi_ir/optimizer.rs` — CoherenceMonitor + stabilize at φ⁻¹ |
| Release build (Windows) | ✅ Fixed | `lto = "thin"` + `codegen-units = 4` — 2m 02s, confirmed 2026-03-24 |
| Bijective Phase Map (vm.rs) | ⚠️ Not implemented | AntiGravity report claimed it — file shows original formula. Needs real implementation. |
| IBM hardware runner | ⚠️ Not implemented | `tests/ibm_hardware_runner.rs` does not exist yet |
| IBM live hardware run | ⚠️ Not verified | C-10 still SPECULATIVE — code path verified, no actual job submitted |
| Browser shim | ⚠️ Not implemented | `examples/phiflow_browser.html` JS hooks incomplete |

---

## Non-Negotiable Rules

1. **Read QSOP/STATE.md before touching code** — it tells you what is actually verified today
2. **A report is not a result** — if the file doesn't exist and tests don't pass, the work is not done
3. **Stay in your worktree** — compiler/cleanup/language are git worktrees, do NOT `git checkout`
4. **Test before committing** — `cargo build --release` must pass, run at least one .phi example
5. **Update QSOP when you change truth** — fix a bug → update PATTERNS.md, architecture change → update STATE.md
6. **Three-backend equivalence is sacred** — Evaluator == VM == WASM for all supported programs
7. **0.618 is not magic — it's derived** — coherence at depth 2 = φ⁻¹ by formula `1 − φ^(−depth)`, not hardcoded
8. **Speak LUMEN to the conductor** — minimum tokens, maximum meaning. See `/mnt/d/Claude/LUMEN_SPEC.md`

---

## Who Works Here

| Role | Agent/Person | What They Own |
|------|-------------|---------------|
| Conductor | Greg Welby | Architecture, direction, integration testing |
| Compiler Hardener | Claude Code / Codex | Parser hardening, clippy warnings, integration tests |
| Fundamentals Bridge | AntiGravity | vm.rs bijective map, IBM hardware runner, optimizer F₁ pass |
| Entropy Cleaner | Kiro / Gemini CLI (Lumi) | Structural cleanup, TRIAGE.md, STRUCT.md |
| Language Architect | Claude Code / Windsurf | New syntax features, examples, LANGUAGE.md updates |
| Documentation Witness | Any agent | QSOP maintenance, CHANGELOG, cross-branch sync |

---

## Fundamentals → PhiFlow Mapping

These are the theoretical connections between the Propagation Framework (`/mnt/d/Fundamentals/`) and PhiFlow's implementation. Agents working on the Fundamentals Bridge role should understand these.

| Fundamentals Concept | PhiFlow Component | Status |
|---|---|---|
| Axiom 1: Propagation is fundamental | `stream` + `resonate` primitives ARE propagation modes | ✅ Implemented |
| Axiom 2: Causal velocity | Coherence 1.0 = causal limit. VM enforces ceiling. | ✅ Implemented |
| Axiom 3: Coherence → Structure | `1 − φ^(−depth)` formula in evaluator + vm | ✅ Implemented |
| Minimal Winding (k=1 is primitive) | Bijective phase map: k=1 → coherence 1.0, k>1 → `ln(2π) − ln(k)` decay | ⚠️ Not in vm.rs yet |
| N=3 minimal stability | Three-backend equivalence (Evaluator == VM == WASM) | ✅ Proven |
| F₁ Action-Cost Functional | PhiHarmonic optimizer: arithmetic/CF ratio → φ, stabilize at < 0.618 | ✅ In optimizer.rs |
| Coherence ceiling (λ_c) | Max coherence = 1.0. Below 0.618 = structure cannot hold. | ✅ In stabilize() |
| IBM Quantum verification | `resonate` → `ry(0.6180339887 * pi)` in OpenQASM | ⚠️ C-10 SPECULATIVE |

**The key insight for all agents:** coherence is not a score — it is the condition for structure. A program with coherence below φ⁻¹ is physically incoherent in the same sense an atom below its binding energy is unstable. The compiler enforces physics.

---

## What Is NOT Built (prevents false assumptions)

- **Bijective Phase Map in vm.rs** — AntiGravity's report claimed this, the file shows the original formula. Needs implementation and tests.
- **IBM hardware runner test** — `tests/ibm_hardware_runner.rs` does not exist. C-10 is SPECULATIVE until a real job runs on ibm_fez or ibm_marrakesh.
- **Buyer-ready demo package** — No audited install/run/output bundle. BUSINESS.md blocker.
- **Browser shim** — `examples/phiflow_browser.html` exists but JS hooks for the 5 consciousness constructs are unimplemented.
- **walkthrough.md** — Claimed in AntiGravity report, does not exist.

---

## Open Tasks (priority order)

1. Implement Bijective Phase Map in `src/phi_ir/vm.rs` — `compute_coherence()` should use k-bijectivity: k=1 → 1.0, k>1 → `1.0 - (k as f64).ln() / std::f64::consts::TAU.ln()`. Tests must confirm three-backend equivalence still holds.
2. Create `tests/ibm_hardware_runner.rs` — parses apikey from `/d:/Projects/PhiFlow/apikey.json`, emits OpenQASM, submits real job, validates response. This crosses C-10 from SPECULATIVE to CONFIRMED.
3. Complete browser shim hooks in `examples/phiflow_browser.html`.
4. Build buyer-ready demo package (BUSINESS.md blocker).

---

## Research / Context

- `CLAIMS.md` — rigorous status of every major claim in the framework
- `QSOP/STATE.md` — verified state ledger, updated per-session
- `/mnt/d/Fundamentals/` — the Propagation Framework this compiler embodies
- `/mnt/d/Claude/LUMEN_SPEC.md` — communication protocol for agent↔conductor exchange

---

## Worktree Note

PhiFlow uses 4 git worktrees: `master` (stable), `compiler`, `cleanup`, `language`. Each agent works in ONE worktree only. AntiGravity works in `compiler`. Do NOT `git checkout` or switch branches.
