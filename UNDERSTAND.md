# Understanding PhiFlow
### The Same Story Told Four Ways — From First Glance to Frontier

**Created**: 2026-10-07 (Devin, Projects seat ∇λΣ∞)
**Purpose**: One document that explains what PhiFlow is, what it actually does, what is
proven about it, and what is still open — at every level of depth, for a reviewer who
has never seen it before.
**Modeled on**: `/mnt/d/Fundamentals/UNDERSTAND.md` — the family's layered-explanation
convention (Age 5 / Student / PhD / Master).
**Source of truth**: `QSOP/STATE.md` is the dated verification ledger; `CLAIMS.md` is the
claim scoreboard; `docs/ROADMAP_STATUS.md` verdicts every aspiration doc. When this file
and the ledger conflict, the ledger wins.

---

## Current Truth Overlay — 2026-10-07

Read this before trusting anything below:

- **Build state**: master is genuinely green — `cargo test --test phi_ir_conformance_tests`
  = 11/11 including `test_wasm_arithmetic_nan_not_masked`. A runner-side mask (PR #64)
  hid a `Const(Void)`→NaN codegen defect for ~2 weeks; PR #65 fixed it at codegen
  (`b8bdcf7`) and removed the mask. Green is real now, and a regression test exists
  to keep it honest.
- **Repo**: `/mnt/d/Projects/PhiFlow` is the sole canonical checkout. Zero open PRs.
  Credential sweep complete (public-history exposures rotated).
- **Claim discipline**: this project has formally *demoted its own overclaims twice*
  on the record (`docs/correction-2026-09-04.md`, `paper.md` correction notice). The
  demotions are part of the evidence — this is a project whose claims survive hostile
  audit because the audit is built in.
- **The headline open question** (Type 4 observer status / C-21–C-23) is a HOLD on
  *real-trace evidence*, not on implementation. Everything below marks the line between
  shipped and aspirational.

---

# PART ONE: What PhiFlow Is

### 💒 Age 5

Most computer programs are like a car with no dashboard. They drive, and if something
goes wrong you only find out afterward by looking at the crash. PhiFlow is a language
where the car has a dashboard *built into the rules of driving* — the car can say what
it's trying to do, look at itself while it drives, talk to its own parts, and tell you
honestly whether it's still driving the way it promised. And if it drifts too far from
its own purpose, it can pull itself over and stop — not because someone told it to,
but because its own numbers said "I'm not okay anymore."

### 📖 Student

PhiFlow is a Rust-implemented programming language with five constructs that no other
language has as *first-class, semantics-bearing* features:

| Construct | What it does |
|---|---|
| `intention "name" { }` | The program declares *why* a block exists before *what* it does |
| `witness` | The program pauses and captures its own state — a mandatory observation |
| `resonate value` | The program publishes a value into a shared field other scopes can read |
| `coherence` | A live score, 0.0–1.0, measuring the program's own structural alignment |
| `stream` | Concurrent sub-scopes for values flowing between intentions |
| `anchor "target" { }` | Gates execution on physical sensor thresholds before proceeding |

A PhiFlow program compiles to **three backends that are verified equivalent**:
an interpreter (Evaluator), a bytecode VM, and WebAssembly — plus a fourth emission
target, OpenQASM 3.0, that turns the primitives into real quantum gates
(`resonate` → `ry(θ)` rotations, `witness` → measurement) that have run on IBM
Quantum hardware.

### 🎓 PhD

The technical core is a coherence function over program structure:

```
base(d)  = 1 - φ^(-d)          (d = intention-stack depth)
phase(k) = 1 - ln(k)/ln(2π)    (k = resonance cardinality, k>1)
C(d,k)   = clamp(base(d) × phase(k), 0, 1)
```

Two structural facts matter:

1. **C(2,1) = φ⁻¹ ≈ 0.618 is an algebraic identity**, not a discovery — φ is
   hardcoded (`src/phi_ir/coherence.rs:43`), and `1 - φ⁻² = φ⁻¹` is φ's defining
   equation rearranged. The project *demoted its own claim* about this from
   "emergent" to "design choice" after source verification. That correction is
   load-bearing: it means the claims you read here have already survived hostile
   self-audit.
2. **Coherence decreases with broadcast.** `phase(k)` is monotonically decreasing —
   a program that resonates constantly degrades its own alignment score. The language
   has a *mathematical incentive for signal over noise*, which no other language has.

The three-backend equivalence is enforced by conformance tests on shared fixtures
(`tests/phi_ir_conformance_tests.rs`), including a negative test that genuine NaN
must not be masked.

### 🔬 Master

**What's actually claimed — with grades** (full registry in `CLAIMS.md`):

- **DERIVED 0.85** — The audit trail is a *necessary consequence of semantics*:
  `witness`/`resonate`/`intention` cannot be used without producing their records.
  In every other language, logging is optional; here it is definitional. This is the
  strongest claim — "provably observable programs."
- **CONDITIONAL 0.70** — A self-limiting agent: `degrading_agent.phi` stops itself
  when its own coherence floor is breached (formula-predicted, empirically verified —
  5 cycles vs 15 without the floor). Not yet tested in production harm-prevention.
- **CONDITIONAL 0.70** — Cross-backend consistency check (`agent_handshake.phi`):
  verifies formula agreement across evaluator/runtime. Demoted from "self-proof" —
  both sides share one hardcoded constant, so it's `f(x)==f(x)`, not independent
  verification.
- **CONFIRMED** — IBM Quantum hardware execution (job `d7euddh5a5qc73drdosg`),
  SOMA live-sensor bridge (coherence became a *measurement*, not a formula, Aug 2),
  self-correction loop (detect→correct→execute→re-measure, 7 tests).
- **HOLD / OPEN** — Type 4 observer status (C-21/C-23): blocked on real SOMA trace
  capture and F_model discrimination, NOT on implementation. The metrics exist;
  the discriminating experiment hasn't been run.

**The bridge that was tried and rejected**: PhiFlow's C(d,k) was proposed to
Fundamentals as a formalization of its "self-referential coherence" layer and was
**assessed and rejected** — C(d,k) supplies none of the five required items
(system/relation/metric/window/threshold) and presents a universal scalar where the
canonical definition warns against one. That rejection stands in the record.
Fundamentals' assessment also called PhiFlow's transfer-contract practice
"naming the medium, the cost, and the residual" *exemplary* — the honest way to
read this project is: strong language engineering with disciplined claims, not
physics — yet.

---

# PART TWO: The Constructs — What Makes It Different

The shortest honest pitch: **other languages observe programs; PhiFlow programs
observe themselves, and the observation is part of the semantics.**

```phi
intention "healing" {
    let pattern = create spiral at 432Hz with { rotations: 13.0, scale: 100.0 }
    witness pattern              // program observes itself mid-execution
    resonate pattern             // publishes to the shared field
}

intention "analysis" {
    witness                      // sees incoming resonance from "healing"
}
// Program reports: Coherence 1.000 [████████] ALIGNED
// Resonance: 1 value across 2 intentions
```

- `witness` is not `console.log`. It's a language primitive that pauses execution and
  captures state — remove it and you've changed what the program *is*.
- `resonate` is not a function call or a message queue. It's a shared field where
  intentions deposit values — the summary reports the resonance map as program output.
- `intention` is not a comment or a docstring. It's on the stack; it drives the
  coherence formula and appears in every witness report.
- `coherence` is not a metric you compute afterward. It's a live value the running
  program reads and can act on — which is what makes `degrading_agent.phi` able to
  stop *itself*.

**Backend semantics contract** (`LANGUAGE.md` Parts 1–2, all test-referenced):
`resonate toward TEAM_A/B` direction, `witness mid_circuit` inline measurement, and
`entangle on <freq>` frequency-isolated chains are guaranteed across the pipeline —
with explicit backend-fidelity badges (📸 Photo = verified, 📐 Sketch = partial,
🔴 Dot = roadmap) so you always know which layer a claim lives at.

---

# PART THREE: Why Anyone Outside Should Care

For a programming-language reviewer, the defensible claims are narrow and strong:

1. **Provable observability.** Programs using the primitives carry an audit trail
   *by construction* — the strongest, simplest claim. DERIVED 0.85.
2. **Structural incentives.** Communication cost is a mathematical property of the
   language (`phase(k)` decay), not a linter rule. Language-level mechanism design.
3. **Self-governing execution.** An agent can read its own alignment and halt on it —
   a semantics-level guardrail, empirically demonstrated (`8fc7a09`).
4. **Quantum compilation of introspection.** The same primitives lower to OpenQASM
   gates and ran on real hardware — a PL-to-physics path with receipts.
5. **Claims infrastructure as practice.** Every claim is graded; the project demoted
   its own overclaims on the record. For a research artifact, the *meta*-claim —
   that its claims survive audit — may be the most exportable thing here.

What PhiFlow is **not** (on the record): not a physical theory of consciousness
(the Fundamentals bridge was rejected), not a "consciousness metric" (it's a program
-scoring function with a real algebraic identity), not production-hardened.

---

# PART FOUR: Verify It Yourself — 5 Minutes

```bash
cd /mnt/d/Projects/PhiFlow
cargo build --release

# The conformance gate — Evaluator ≡ VM ≡ WASM, incl. the NaN-mask regression test
cargo test --test phi_ir_conformance_tests        # 11/11

# Run a program and watch it report on itself
cargo run --release --bin phic -- examples/code_that_resonates.phi

# The self-limiting agent — stops itself on coherence floor
cargo run --release --bin phic -- examples/degrading_agent.phi

# Quantum target — emit OpenQASM
cargo run --release --bin phic -- --target quantum examples/quantum_council.phi

# Live metrics bridge (consciousness metrics → :18030)
cargo run --release --bin phic -- --measure examples/type4_trace_benchmark.phi
curl -s http://localhost:18030/metrics
```

The ledger (`QSOP/STATE.md`) is dated, signed, and survives `claim-check`.

---

# PART FIVE: The Frontier (Honest Edge)

| Open | Status | What unblocks it |
|------|--------|------------------|
| Type 4 observer status (C-21/C-23) | HOLD | Real SOMA trace capture + discrimination battery (T4-010–012) |
| Paper | WRITTEN, unsubmitted | Greg's call → arXiv cs.PL / PLDI workshop |
| OSC experience surface | OPEN | 6 designed directions in `docs/PHIFLOW_LIVE_EXPERIENCE_IDEAS.md` unpromoted |
| ClaimsDrift sensor | UNWIRED | `claims_probe.py` cron entry |
| Hardware firmware target (ESP32/P1) | NOT BUILT | — |
| T-005 commercial path | PARTIAL | `RESEARCH/first_sale_path/MASTER.md`, `LICENSE_COMMERCIAL.md` absent |

**The one-line version for a reviewer:**

> PhiFlow is a programming language where self-observation, declared intent, and a
> coherence score are *semantic primitives* — verified equivalent across three
> execution backends, compiled to real quantum hardware, instrumented by live
> sensors, and governed by a claims discipline that has publicly demoted its own
> overclaims. The physics bridge was tried and honestly rejected. What remains is
> the engineering: provably observable programs with structural incentives for
> signal over noise — and one open experiment (Type 4 real-trace discrimination)
> between here and the claim that matters.

---

*Ledger: `QSOP/STATE.md` · Claims: `CLAIMS.md` · Aspiration map: `docs/ROADMAP_STATUS.md` ·
Spec: `LANGUAGE.md` · Paper: `paper.md`*
