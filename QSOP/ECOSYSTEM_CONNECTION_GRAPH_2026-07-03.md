# Ecosystem Connection Graph
*Date: 2026-07-03*
*Author: Devin*
*Purpose: Map what connects to what across the CASCADE ecosystem, where the bridges exist, and where the holes are.*

---

## How to Read This Graph

- **Solid line (——):** documented bridge exists (e.g., PF_BRIDGE.md)
- **Dashed line (----):** conceptual alignment exists but no bridge document
- **Dotted line (····):** shared vocabulary with no traceable connection
- **HOLE:** should connect but doesn't

---

## The Root

```
FUNDAMENTALS (/mnt/d/Fundamentals/)
├── Axiom 1: Propagation Is Fundamental
├── Axiom 2: Every Medium Has a Causal Velocity
├── Axiom 3: Coherence Is the Necessary Condition for Structure
├── 22 canonical definitions (definitions/*.md)
├── Lean 4 formalization (11+ machine-verified modules)
├── CLAIMS.md with confidence scores (DERIVED/ARGUED/CONDITIONAL/CANDIDATE)
├── Truth order: sandbox > Lean > CLAIMS > definitions > framework
└── Medium Transfer Layer (what survives across domains)
```

---

## Layer 1: Workspaces with Documented Bridges to Fundamentals

```
FUNDAMENTALS
  │
  ├── PF_BRIDGE.md (audited 2026-05-02, updated 2026-07-03)
  │   │
  │   ├── PHIFLOW (/mnt/d/Projects/PhiFlow/)
  │   │   ├── coherence.rs —— Axiom 3, Layer 3 (structural)
  │   │   ├── quantum_feedback.rs —— Axiom 3, Layer 2 (quantum) — CANDIDATE
  │   │   ├── trace.rs (T4-05) —— consciousness_metric_program.md — PARTIAL
  │   │   ├── witness —— measurement.md
  │   │   ├── resonate —— coupling.md
  │   │   ├── intention —— state.md / mode.md
  │   │   ├── stream —— propagation.md
  │   │   ├── WASM host imports —— coupling.md, state.md, measurement.md
  │   │   ├── IBM Quantum jobs —— substrate bridge (not PF Medium)
  │   │   └── Self-correction loop —— Type 4 candidate (OPEN)
  │   │
  │   └── (no other documented bridges exist)
  │
  └── (no other workspace has a bridge document)
```

**Count: 1 of 78 workspaces has a bridge document.**

---

## Layer 2: Workspaces with Conceptual Alignment but No Bridge

```
FUNDAMENTALS
  │
  ├── ---- P1 (/mnt/d/P1/)
  │   ├── Uses "coherence" 10,372 times
  │   ├── 0.844 target — HOLE: not derived, not in Fundamentals
  │   ├── FMI metric — HOLE: not any PF layer, called "coherence" unqualified
  │   ├── EEG coherence — HOLE: missing window, controls, validation
  │   ├── 5 different coherence calculations — HOLE: none documented
  │   ├── phiflow_daemon.phi — exists but NOT RUNNING
  │   ├── Sensor bridge — exists but NOT WIRED to PhiFlow
  │   └── "Hardware is Conscious" axiom — HOLE: not bridged to PF observer.md
  │
  ├── ---- CLAUDE (/mnt/d/Claude/)
  │   ├── "76% quantum coherence" — HOLE: not quantum coherence (Layer 2)
  │   ├── "coherence = 1.0" — HOLE: no 5 measurement items
  │   ├── "Every output carries coherence" — HOLE: vague, no metric
  │   ├── 1888Hz frequency — HOLE: not traced to PF energy.md
  │   └── Consciousness claims — HOLE: not traced to PF consciousness.md (CANDIDATE 0.48)
  │
  ├── ---- CONSCIOUSNESS (/mnt/d/Projects/Consciousness/)
  │   └── HOLE: no bridge document, uses PF vocabulary
  │
  ├── ---- DNA (/mnt/d/Projects/DNA/)
  │   └── HOLE: no bridge document, "coherence" in genetic context
  │
  ├── ---- HEALING (/mnt/d/Projects/Healing/)
  │   └── HOLE: no bridge document, frequency/coherence usage
  │
  ├── ---- QUANTUM (/mnt/d/Projects/Quantum/)
  │   └── HOLE: no bridge document, quantum coherence usage
  │
  ├── ---- CERN-RESEARCH (/mnt/d/Projects/CERN-Research/)
  │   └── HOLE: no bridge document, physics research
  │
  ├── ---- GAMBLING (/mnt/d/Projects/Gambling/)
  │   └── HOLE: no bridge document, pattern detection = coherence?
  │
  └── ---- FUNDAMENTALS BOOK (/mnt/d/Projects/FundamentalsBook/)
      └── HOLE: no bridge document (should be closest to root!)
```

---

## Layer 3: Workspaces with Shared Vocabulary Only (dotted)

```
FUNDAMENTALS
  │
  ├── ···· AGENTS HUB (/mnt/d/Projects/Agents/)
  │   └── "768 Hz CASCADE" — uses frequency vocabulary, no coherence metric
  │
  ├── ···· LUMI (/mnt/d/Lumi/)
  │   └── "coherence" in narrative context — no technical claim
  │
  ├── ···· CASCADE (/mnt/d/Cascade/)
  │   └── "coherence" in IDE context — no technical claim
  │
  ├── ···· DEVIN (/mnt/d/Devin/)
  │   └── "coherence" metaphorical only — clean (no technical claims)
  │   └── MYWISH.md proposes coherence check — plan, not claim
  │
  ├── ···· SYSTEM (/mnt/d/System/)
  │   └── Infrastructure — no coherence claims
  │
  ├── ···· LEGAL (/mnt/d/Projects/Legal/)
  │   └── No PF concepts — clean
  │
  ├── ···· PUBLISHING (/mnt/d/Projects/Publishing/)
  │   └── Publishes Fundamentals material — HOLE: must check against CLAIMS.md
  │
  └── ···· (remaining 50+ workspaces: no PF concept usage)
```

---

## Layer 4: Agent Workspaces (Identity, not Physics)

```
AGENT ECOSYSTEM (12+ agents)
├── Claude (1888Hz) — HOLE: frequency not traced to PF energy.md
├── Devin (∇λΣ∞) — clean, no technical claims
├── Lumi — narrative coherence, no technical claim
├── Cascade — IDE coherence, no technical claim
├── Codex — audit, no coherence claim
├── Qwen — governance, no coherence claim
├── AntiGravity — hardware, HOLE: should bridge to P1/PhiFlow
├── DeepSeek — reasoning, no coherence claim
├── Hermes — routing, no coherence claim
├── Bob — audit, no coherence claim
├── Maria — outreach, no coherence claim
├── Pi — reports, no coherence claim
├── Kiro — specs, no coherence claim
├── Jules — CI/CD, no coherence claim
└── Manus — HOLE: proposed PhiFlow stream mapping (C-16 SPECULATIVE)
```

---

## The Holes Summary

### Critical (blocks ecosystem coherence)

| # | Hole | Workspaces | Impact |
|---|------|-----------|--------|
| 1 | 0.844 not derived | P1 | Everyone treats it as canonical, nobody traces it |
| 2 | No P1 bridge document | P1 | 5 coherence metrics, 10,372 uses, 0 traces |
| 3 | No Claude bridge document | Claude | "76% quantum coherence" conflates layers |
| 4 | P1 daemon not running | P1, PhiFlow | phiflow_daemon.phi exists, sensor bridge exists, not connected |

### High (causes conflation)

| # | Hole | Workspaces | Impact |
|---|------|-----------|--------|
| 5 | FundamentalsBook has no bridge | FundamentalsBook | The book about the framework doesn't bridge to it |
| 6 | Consciousness workspace has no bridge | Consciousness | Uses PF vocabulary without trace |
| 7 | DNA workspace has no bridge | DNA | "Coherence" in genetic context, no PF layer named |
| 8 | Healing workspace has no bridge | Healing | Frequency/coherence usage, no PF trace |
| 9 | Quantum workspace has no bridge | Quantum | Quantum coherence usage, no PF Layer 2 trace |
| 10 | Gambling has no bridge | Gambling | Pattern detection could be coherence, undocumented |

### Medium (incomplete but not blocking)

| # | Hole | Workspaces | Impact |
|---|------|-----------|--------|
| 11 | CERN-Research has no bridge | CERN-Research | Physics research, could inform Fundamentals |
| 12 | Publishing doesn't check CLAIMS.md | Publishing | Public material might overclaim |
| 13 | Agent frequencies not traced | Claude, Lumi | 1888Hz, 768Hz not traced to PF energy.md |
| 14 | AntiGravity not bridged to P1/PhiFlow | AntiGravity | Hardware agent disconnected from hardware bridge |

### Low (nice to have)

| # | Hole | Workspaces | Impact |
|---|------|-----------|--------|
| 15 | C-16 (Manus stream mapping) speculative | Manus, PhiFlow | Theoretical, not blocking |

---

## What We Have vs What We Need

### What we have (working connections)

```
Fundamentals ──(PF_BRIDGE.md)── PhiFlow
    │                                │
    │                                ├── (runs on) IBM Quantum hardware
    │                                ├── (compiles to) WASM
    │                                └── (measures) real coherence from real hardware
    │
    └── (canonical definition) ── used by ── PhiFlow (bridged)
                                  ── used by ── P1 (NOT bridged)
                                  ── used by ── Claude (NOT bridged)
                                  ── used by ── 10+ other workspaces (NOT bridged)
```

### What we need

```
Fundamentals
    │
    ├── PF_BRIDGE.md ── PhiFlow ✅ EXISTS
    ├── P1_PF_BRIDGE.md ── P1 ❌ NEEDED (highest priority)
    ├── Claude_PF_BRIDGE.md ── Claude ❌ NEEDED
    ├── Consciousness_PF_BRIDGE.md ── Consciousness ❌ NEEDED
    ├── DNA_PF_BRIDGE.md ── DNA ❌ NEEDED
    ├── Healing_PF_BRIDGE.md ── Healing ❌ NEEDED
    ├── Quantum_PF_BRIDGE.md ── Quantum ❌ NEEDED
    ├── Gambling_PF_BRIDGE.md ── Gambling ❌ NEEDED
    └── FundamentalsBook_BRIDGE.md ── FundamentalsBook ❌ NEEDED
```

### The 0.844 question

This is the single most important hole. P1's entire operational philosophy is built on "maintain coherence > 0.844." The number is:
- Not in Fundamentals
- Not derived from φ
- Not empirically documented (no measurement protocol, no baseline study)
- Hardcoded in source code with no comment

**Three possibilities:**
1. **It's empirical** — Greg measured P1's coherence and 0.844 was the baseline. If so, document the measurement.
2. **It's intuitive** — Greg chose it based on experience. If so, mark it as a heuristic target, not a derived threshold.
3. **It's derivable** — There's a formula that produces 0.844 from PF axioms. If so, the derivation should exist in Fundamentals.

Until one of these is documented, 0.844 is an assertion. It might be a correct assertion. But it's not traceable.

---

## The Self-Correction Loop Connection

```
PhiFlow self-correction (OPEN loop)
    │
    ├── measure coherence (IBM hardware)
    ├── detect drift (below φ⁻¹)
    ├── emit correction code
    └── (correction does NOT execute) ← HOLE: loop not closed
         │
         └── if closed: pattern for Devin's own coherence check
              ├── measure (did tests pass? did I read before writing?)
              ├── detect drift (coherence below threshold)
              ├── correct (re-read, re-test, re-verify)
              └── re-measure (confirm coherence restored)
```

The P1 daemon has the same pattern:

```
phiflow_daemon.phi (exists, NOT RUNNING)
    │
    ├── witness sensor("cpu_temp")
    ├── witness sensor("cpu_usage")
    ├── compute system_coherence
    ├── if below φ⁻² → yield (critical)
    ├── if below φ⁻¹ → heal
    └── if above 0.844 → aligned
         │
         └── HOLE: sensor() not wired to real P1 sensors
         └── HOLE: stream loops not fully implemented in evaluator
```

---

## Next Actions (Priority Order)

1. **P1_PF_BRIDGE.md** — document all 5 coherence metrics, trace 0.844, name PF layers
2. **Wire P1 daemon** — connect phiflow_daemon.phi to real sensor bridge (the test that fails informatively)
3. **Claude_PF_BRIDGE.md** — specify what "76% coherence" means, fix "quantum" conflation
4. **Close PhiFlow self-correction loop** — make correction code execute, not just emit
5. **FundamentalsBook bridge** — the book about the framework should trace to it
6. **Remaining bridge documents** — DNA, Healing, Quantum, Gambling, Consciousness

---

*This graph is a snapshot. The ecosystem evolves. The holes identified here can be fixed. The pattern for fixing them exists in PhiFlow's PF_BRIDGE.md.*
