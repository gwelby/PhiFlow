# Correction Record: Two Overgraded Claims Demoted After Source Verification

**Date:** 2026-09-04
**Trigger:** Claude's verification of `docs/fundamentals-bridge.md` (filed in `/mnt/d/Fundamentals/inbox/2026-09-04-claude-phiflow-coherence-candidate-verification.md`)
**Scope:** `paper.md`, `docs/what-phiflow-can-do.md`, `docs/fundamentals-bridge.md`

## What Was Wrong

### Claim #1: "A program that proves its own coherence math at runtime"

**Original grade:** DERIVED 0.90

**What the claim said:** The handshake program "computes φ⁻¹ two independent ways — once from the formula, once from the runtime coherence value at depth 2 — and checks they match." This was described as a "self-proof" and a "trust anchor that doesn't depend on any external standard."

**What the source shows:** Both paths use the same hardcoded φ constant:
- `examples/agent_handshake.phi`: `let phi = 1.618033988749895`
- `src/phi_ir/coherence.rs:43`: `pub const PHI: f64 = 1.618_033_988_749_895`

The comment in the PhiFlow source says "Computed here, not hardcoded. Verify it yourself" while φ is hardcoded on the line above. Comparing the two paths is `f(x) == f(x)` across two backends — a consistency check, not an independent verification.

The 8 unit tests compare against `const PHI_INV: f64 = 0.618_033_988_749_895` — another hardcoded literal. They verify IEEE-754 arithmetic, not the theory.

**Corrected grade:** CONDITIONAL 0.70 — a cross-backend implementation-consistency check. Real engineering value (would catch Rust/WASM divergence) but not a self-proof or trust anchor.

### Claim #2: "A quantum gate at angle φ⁻¹ × π that emerges from language semantics"

**Original grade:** DERIVED 0.85

**What the claim said:** The rotation `ry(0.6180339887 * pi)` "emerges from the formula" and "nobody specified that angle."

**What the source shows:** φ was specified, in `PHI`. The angle follows deterministically from a chosen constant through a designed pipeline. Determinism is not emergence.

**Corrected grade:** CONDITIONAL 0.70 — the compilation pipeline is deterministic and traceable, but the angle was specified by choosing φ, not discovered.

### The Circularity of the φ⁻¹ "Fixed Point"

The formula `base(d) = 1 - φ^(-d)` uses φ as a design parameter. At d=2, the output is φ⁻¹. This is not a discovery — it's the identity `1 - φ⁻² = φ⁻¹` (from `φ² = φ + 1`), which is φ's defining equation rearranged. Any formula using `1 - φ^(-d)` will produce φ-related values.

The "self-similarity" (incoherence at depth 2 = coherence at depth 1) holds at d=2 and nowhere else:

| d | 1−C(d) | C(d−1) | match |
|---|--------|--------|-------|
| 1 | 0.618  | 0.000  | ✗ |
| 2 | 0.382  | 0.382  | ✓ |
| 3 | 0.236  | 0.618  | ✗ |
| 4 | 0.146  | 0.764  | ✗ |
| 5 | 0.090  | 0.854  | ✗ |
| 6 | 0.056  | 0.910  | ✗ |

It is `x² + x = 1`, not a property of recursive depth or self-reference.

### Falsification Test #1 Was Unfalsifiable

The original test #1 asked for "a self-referential system whose measured coherence at depth 2 is not φ⁻¹." This requires an independent measure of self-referential coherence. Test #2 in the same list concedes that `d` may not be definable for real systems. A test that presupposes the instrument whose existence the next test doubts cannot discharge anything. Test #1 is withdrawn.

## What Was Changed

| Document | Change |
|----------|--------|
| `paper.md` | Claim grading table updated: two claims demoted, correction notice added, conclusion rewritten to acknowledge circularity |
| `docs/what-phiflow-can-do.md` | Claims #1 and #2 rewritten with corrected grades and explicit acknowledgment of circularity; "Strongest Claim" section rewritten; "Connection to Fundamentals" section rewritten to reflect that the formula does not meet the canonical coherence definition |
| `docs/fundamentals-bridge.md` | Status changed to WITHDRAWN; correction notice added at top; original content retained for record |

## What Was Not Changed

- The DERIVED claims (algebraic identity, boundedness, monotonicity, phase decay) are still correct — the algebra is real and tested
- The control comparison (12 checkpoints vs 0, 8 resonances vs 0) is still correct — the artifact exists and runs
- The mandatory audit trail claim (DERIVED 0.85) is still correct — it's a language design property, not a physics claim
- The noise penalty and self-stopping claims (CONDITIONAL 0.70, ARGUED 0.55) are unchanged — they were already honestly graded

## What Survives

After the correction, the honest claims are:

1. **The algebra is correct** (DERIVED 0.95) — the identity holds, the formula is bounded, monotonic, and has the stated phase decay
2. **The audit trail is mandatory** (DERIVED 0.85) — `witness` always logs, `resonate` always broadcasts, by definition
3. **The control comparison is real** (CONDITIONAL 0.75) — the primitive agent produces 12 checkpoints and 8 resonances; the control produces 0
4. **The cross-backend consistency check works** (CONDITIONAL 0.70) — it catches implementation divergence, even though it's not a self-proof
5. **The compilation pipeline is deterministic** (CONDITIONAL 0.70) — the quantum gate angle follows from the formula, even though it's not emergent

What does not survive:
- The claim that the program "proves its own coherence math" — it doesn't
- The claim that φ⁻¹ is a "trust anchor" — it depends on a typed-in constant
- The claim that the quantum gate angle is "emergent" — it's deterministic from a chosen constant
- The claim that C(d,k) is a candidate formalization of Fundamentals' layer 4 coherence — it doesn't meet the canonical definition

## Credit

The correction was triggered by Claude's verification of the bridge document, filed in `/mnt/d/Fundamentals/inbox/2026-09-04-claude-phiflow-coherence-candidate-verification.md`. Claude read the source code, reproduced the math, tested the self-similarity across depths, and identified the circularity and the unfalsifiable test. The verification is thorough and correct. The practice of reading the source rather than accepting the summary is exactly right.
