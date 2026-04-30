# Coherence Layer Specification for PhiFlow
*Last updated: 2026-04-30*
*Status: AUDITED DRAFT*

**Purpose:** Map PhiFlow's `coherence` construct to PF coherence layers per `D:\Fundamentals\definitions\coherence.md`.

---

## PF Coherence Layers (from `coherence.md`)

| Layer | Meaning | Math Object | Example |
|-------|---------|-------------|---------|
| 1 Phase/Wave | Stable phase relation | First-order coherence function | Laser light |
| 2 Quantum | Superposition structure | Off-diagonal density-matrix terms | Qubit superposition |
| 3 Structural/Dynamical | Persistent correlations under evolution | Eigenmodes, attractors | Stable particles |
| 4 Self-referential | System models own state | Not yet formalized | Consciousness candidates |

---

## PhiFlow Coherence: What Layer?

**Primary layer: Structural/Dynamical analogue (Layer 3)**

Evidence:
- `canonical_coherence(depth, k)` measures a PhiFlow runtime stability proxy under evaluation
- Formula `base * phase` is domain-specific (evaluator dynamics), not quantum density-matrix
- 0.618 at depth 2 with `k <= 1` is a confirmed PhiFlow runtime invariant, not a PF-derived threshold

**Secondary involvement: Layer 2 (Quantum)**

Evidence:
- When `--target openqasm`, coherence emits `ry(0.618 * pi)` — **quantum gate**
- IBM hardware run verifies a physical quantum execution path for generated OpenQASM
- But: coherence *value itself* is classical scalar, not quantum superposition measure

**NOT Layer 1 (Phase/Wave):** No oscillatory phase comparison in PhiFlow coherence
**NOT Layer 4 (Self-referential):** No self-model loop in coherence formula itself; daemon provides this at higher level

---

## Measurement Discipline

Per PF requirement, PhiFlow coherence specifies:

| Discipline | PhiFlow Answer |
|------------|---------------|
| **System** | Evaluator state + optional sensor readings + optional quantum register target |
| **Relation** | Multiplicative correlation between base intent and phase alignment |
| **Metric** | `base * phase` (f64 scalar, 0.0-1.0) |
| **Window** | Single evaluation step / daemon tick |
| **Threshold** | 0.618 at depth = 2, k <= 1 (C-3 in CLAIMS.md); not PF-derived |

---

## Bridge Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| PhiFlow coherence is a Layer 3 structural analogue | ✅ SUPPORTED | `src/phi_ir/coherence.rs` + evaluator persistence |
| PhiFlow coherence can realize Layer 2 | ✅ SUPPORTED | OpenQASM emission + IBM hardware verification |
| PhiFlow coherence is sufficient for Layer 4 | 🔬 PARTIAL | Daemon adds self-reference; coherence alone does not |
| PF "coherence ceiling" has PhiFlow formula | ❌ NOT DERIVED | No PF derivation of 0.618 threshold from first principles |

---

## Open Questions

- Does PF imply PhiFlow's `base * phase` as canonical structural coherence functional? **OPEN**
- Can Layer 4 self-referential coherence be defined on top of PhiFlow daemon state? **OPEN**
- Should PhiFlow expose layer selection (`coherence quantum` vs `coherence structural`)? **OPEN**

---

*Status: AUDITED DRAFT — safe if kept as PhiFlow-specific layer mapping, not PF derivation*
