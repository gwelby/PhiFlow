# AGENTS.md: PhiFlow
*Platform / Tool (Type 2) + Theory / Research (Type 3)*
*Last updated: 2026-03-29 truth-sync baseline*

**Master AGENTS.md:** `D:\Projects\PhiFlow\AGENTS.md`
**This worktree:** compiler — AntiGravity works here. `src/phi_ir/vm.rs`, `tests/ibm_hardware_runner.rs`
**Communication**: LUMEN → `/mnt/d/Claude/LUMEN_SPEC.md`
**Operations**: QSOP → `/mnt/d/Claude/QSOP_SPEC.md`

## Open Questions
Single blocker: **IBM live hardware run blocked** — `tests/ibm_hardware_runner.rs` gets 403 auth error on `GET /v1/backends`. Until a successful IBM receipt exists, C-10 stays speculative. Browser host (canonical coherence math) is secondary.

---

## Mission

PhiFlow is a Rust compiler and runtime for consciousness-aware programming.

Programs written in PhiFlow have first-class operations to name intentions, witness their own state, measure coherence, and resonate values to other programs. The runtime paths in this checkout target native execution, bytecode, WebAssembly host imports, and OpenQASM 3.0 emission.

The deeper mission: PhiFlow makes the Propagation Framework executable. Every program is a coherence experiment. The compiler is a physics instrument.

---

## Why PhiFlow Exists

Three things converged:

1. **The Fundamentals framework** derived that coherence is the necessary condition for structure. In the current runtime, canonical coherence is multiplicative: `base(depth) * phase(k)`. At depth 2 with `k <= 1`, it yields `phi^-1 = 0.618033988749895`.
2. **IBM Quantum hardware** accepts OpenQASM 3.0 programs. The bridge between PhiFlow semantics and physical quantum circuits is a compiler/backend problem, not a philosophical claim.
3. **No language existed** that treated coherence as a first-class computational resource. PhiFlow is that language.

---

## Truth Order

When files disagree, trust in this order:

1. Running code and test results (`cargo test` output, verified in `QSOP/STATE.md`)
2. `QSOP/STATE.md` — dated verification ledger
3. `WORKSPACE.md` — technical state summary
4. `CLAIMS.md` — research claim status
5. `TASKS.md` — work queue
6. `README.md`, `VISION.md`, narrative docs — aspirational or historical context

*Running tests beat the spec.*
*A report claiming work was done means nothing without the file existing and tests passing.*

---

## Current State

| Component | Status | Notes |
|-----------|--------|-------|
| Parser | ✅ Verified | `src/parser/mod.rs` — handles the five core constructs |
| PhiIR + Lowering | ✅ Verified | `src/phi_ir/` — SSA IR with consciousness nodes |
| Evaluator (reference) | ✅ Verified | `src/phi_ir/evaluator.rs` delegates coherence to `src/phi_ir/coherence.rs` |
| PhiVM (bytecode) | ✅ Verified | `src/phi_ir/vm.rs` delegates coherence to `src/phi_ir/coherence.rs` |
| Canonical coherence module | ✅ Verified | `src/phi_ir/coherence.rs` shared by evaluator, VM, and `tests/phi_ir_wasm_runner.js` |
| WASM codegen | ✅ Verified | `src/phi_ir/wasm.rs` — native WASM path covered by conformance tests and host runner |
| WASM host bridge | ✅ Verified | `src/wasm_host.rs` — `wasmtime` bridge with hook callbacks |
| MCP server | ✅ Verified | `src/bin/phi_mcp.rs` — spawn/resume/read, shared resonance |
| Sensors | ✅ Verified | `src/sensors.rs` — CPU/memory/thermal/network via `sysinfo` |
| PhiHarmonic optimizer | ✅ Verified | `src/phi_ir/optimizer.rs` — stabilize at `phi^-1` threshold |
| Release build (Windows) | ✅ Confirmed | `lto = "thin"` + `codegen-units = 4`, verified 2026-03-24 in `QSOP/STATE.md` |
| IBM hardware runner | ⚠️ Structurally ready | `tests/ibm_hardware_runner.rs` exists; compile gate is live and the real-hardware test is `#[ignore]` |
| IBM live hardware run | ⚠️ Not verified | 2026-03-29 live gate reached IBM Cloud Runtime and failed `GET /v1/backends` with `403` authorization error |
| Browser host | ⚠️ Experimental | `examples/phiflow_browser.html` implements the five hooks, but still requires manual hosting/build artifacts and uses non-canonical host-side coherence math |

---

## Non-Negotiable Rules

1. **Read `QSOP/STATE.md` before touching code** — it tells you what is actually verified today
2. **A report is not a result** — if the file does not exist and tests do not pass, the work is not done
3. **Stay in your worktree** — compiler/cleanup/language are git worktrees, do not `git checkout`
4. **Test before committing** — `cargo build --release` must pass, and run at least one `.phi` example when the shell supports Cargo
5. **Update QSOP when you change truth** — fix a bug -> update the relevant ledger entry; architecture change -> update `STATE.md`
6. **Three-backend equivalence is sacred** — Evaluator == VM == canonical WASM runner for supported programs
7. **0.618 is derived** — canonical coherence is `base(depth) * phase(k)`; depth 2 with `k <= 1` yields `phi^-1`
8. **Do not overstate hardware or browser claims** — IBM live execution stays speculative until a successful receipt exists, and browser host claims must match the actual demo semantics

---

## Who Works Here

| Role | Agent/Person | What They Own |
|------|--------------|---------------|
| Conductor | Greg Welby | Architecture, direction, integration testing |
| Compiler Hardener | Claude Code / Codex | Parser hardening, clippy warnings, integration tests |
| Fundamentals Bridge | AntiGravity | IBM runtime path, optimizer/F1 pass, backend proof plumbing |
| Entropy Cleaner | Kiro / Gemini CLI (Lumi) | Structural cleanup, TRIAGE, STRUCT |
| Language Architect | Claude Code / Windsurf | New syntax features, examples, language docs |
| Documentation Witness | Any agent | QSOP maintenance, changelog, cross-branch sync |

---

## Fundamentals -> PhiFlow Mapping

| Fundamentals Concept | PhiFlow Component | Status |
|---|---|---|
| Axiom 1: Propagation is fundamental | `stream` + `resonate` primitives are propagation modes | ✅ Implemented |
| Axiom 2: Causal velocity | Coherence 1.0 is a ceiling, not a default | ✅ Implemented |
| Axiom 3: Coherence -> Structure | `src/phi_ir/coherence.rs` computes canonical multiplicative coherence | ✅ Implemented |
| Minimal Winding proposal | `k = 1 -> 1.0`, `k > 1 -> 1.0 - ln(k) / ln(TAU)` | ⚠️ Superseded proposal, not current repo truth |
| N=3 minimal stability | Evaluator == VM == canonical WASM runner | ✅ Proven |
| F1 Action-Cost Functional | PhiHarmonic optimizer ratio -> stabilize below `0.618` | ✅ Implemented |
| Coherence ceiling (`lambda_c`) | Max coherence = 1.0; below `phi^-1` structure cannot hold | ✅ Implemented |
| IBM Quantum verification | `resonate` lowers to OpenQASM rotation, live run still blocked on auth | ⚠️ Structurally ready, not live-confirmed |

**Key insight:** coherence is not a vibe score. It is the runtime condition for structure. Do not collapse canonical multiplicative coherence into stale additive or narrative-only formulas when updating docs or hosts.

---

## What Is NOT Built

- **Confirmed IBM live receipt** — `tests/ibm_hardware_runner.rs` exists, but C-10 remains speculative until the ignored live test succeeds and writes a scrubbed receipt
- **Canonical browser host** — `examples/phiflow_browser.html` exists, but it still uses older host-side coherence math and requires manual generation/serving of `output.wasm` or `output.wat`
- **Buyer-ready demo package** — no audited install/run/output bundle yet

---

## Open Tasks (priority order)

1. Resolve the IBM Cloud authorization boundary for `tests/ibm_hardware_runner.rs`, then rerun the ignored live gate and capture a scrubbed receipt.
2. Canonicalize `examples/phiflow_browser.html` to the shared multiplicative coherence semantics in `src/phi_ir/coherence.rs`, and document the manual browser-host prerequisites.
3. Add a one-command verification gate for the core truth tests (`openqasm`, golden integration, repro bugs, and targeted conformance).
4. Build a buyer-ready demo package that cites only verified capabilities.

---

## Research / Context

- `CLAIMS.md` — rigorous status of major claims
- `QSOP/STATE.md` — verified state ledger, updated per session
- `WORKSPACE.md` — operational summary for this checkout
- `D:\Fundamentals\` — the Propagation Framework this compiler embodies
- `D:\Claude\LUMEN_SPEC.md` — communication protocol for agent <-> conductor exchange

---

## Worktree / Dispatch

Four git worktrees share one history. Each agent works in ONE only.
- **compiler** → `PhiFlow-compiler/AGENTS.md` (AntiGravity/Codex — parser hardening + IBM auth)
- **cleanup** → `PhiFlow-cleanup/AGENTS.md` (Kiro/Lumi — entropy reduction, TRIAGE.md)
- **language** → `PhiFlow-lang/AGENTS.md` (Claude Code/Windsurf — new syntax features)
- **master** → this directory (documentation, QSOP maintenance, merges)

Full dispatch prompts (copy-paste to spin up any agent): `DEPLOY.md`
