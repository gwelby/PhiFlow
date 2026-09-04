# PhiFlow → Fundamentals Bridge: Self-Referential Coherence Candidate

**Purpose:** Pass forward the one mathematical observation from PhiFlow that is relevant to Fundamentals' open question on self-referential coherence.

**Status:** **WITHDRAWN as a candidate for Fundamentals layer 4 coherence.** The formula does not meet the canonical coherence definition's requirements (system, relation, metric, window, threshold). The φ⁻¹ "fixed point" is circular (φ baked into the formula produces φ-related outputs). The self-similarity holds at d=2 only, not generally. The bridge is retained as a record of the submission and the Fundamentals assessment.

**Correction (2026-09-04):** This document was submitted to the Fundamentals inbox and assessed by Claude. The assessment found that:
1. The φ⁻¹ "fixed point" is φ's defining equation rearranged, not a discovery about self-reference
2. The "self-verification" claim (DERIVED 0.90) does not survive the source — both paths use the same hardcoded φ constant
3. The formula does not satisfy Fundamentals' canonical coherence definition (no system, relation, metric, window, or threshold for a physical system)
4. C(d,k) is presented as a universal scalar, which the canonical definition explicitly warns against
5. Falsification test #1 is unfalsifiable as written (requires an independent measure that test #2 admits may not exist)

The assessment's disposition: file as received, do not promote to CLAIMS.md, do not cite in the manuscript. The transfer contract practice was noted as exemplary. The engineering (mandatory audit trails, noise penalties, self-halting) is real language design worth keeping, but it is not physics.

The original document is retained below for the record.

## The Open Question in Fundamentals

`/mnt/d/Fundamentals/definitions/coherence.md` defines four layers of coherence:

1. Phase/wave coherence — formalized (PLV, wPLI, cross-spectrum)
2. Quantum coherence — formalized (off-diagonal density-matrix terms)
3. Structural/dynamical coherence — formalized (eigenmodes, attractors, solitons)
4. Self-referential coherence — **"Not yet formalized for PF"**

Layer 4 is described as: "A speculative extension where a system maintains an integrated model of its own state/process."

## The Candidate from PhiFlow

PhiFlow defines a coherence function C(d, k) that maps a system's structural properties to a bounded score:

```
base(d) = 0              when d = 0
          1 - φ^(-d)     otherwise

phase(k) = 1.0           when k ≤ 1
           1 - ln(k)/ln(τ)  otherwise

C(d, k) = clamp(base(d) × phase(k), 0, 1)
```

Where:
- d = depth of nested self-modeling (how many levels of "I am observing myself observing myself...")
- k = cardinality of self-broadcast (how many times the system has communicated its own state)
- φ = golden ratio, τ = 2π

**The claim:** d and k are structural properties of a self-referential system. d measures how deeply the system models itself. k measures how much the system has broadcast its own state. C(d,k) is a candidate formalization of layer 4 coherence.

## The Mathematical Property That Matters

At d=2, k≤1:

```
C(2, 1) = (1 - φ^(-2)) × 1.0
        = 1 - 1/φ²
        = 1 - 1/(φ+1)     [because φ² = φ + 1]
        = 1 - 0.381966...
        = 0.618033...
        = φ⁻¹
```

**φ⁻¹ is an algebraic fixed point of the coherence function.** It emerges from the formula, not from a tuned constant. This is machine-verified by 8 unit tests in `src/phi_ir/coherence.rs` to 1e-10 precision.

## Why This Is Relevant to Fundamentals

Fundamentals' Axiom 3 says: "Coherence is the necessary condition for structure." The axiom describes coherence as existing on a spectrum from zero (thermal noise) to self-referential (systems whose coherent patterns include models of their own coherence).

If layer 4 (self-referential coherence) has a natural fixed point — a structural configuration where the coherence score is algebraically determined — then φ⁻¹ is a candidate for that fixed point. The argument:

1. Self-referential coherence requires the system to model itself. This is depth d ≥ 1.
2. At d=1, the system models itself once. Coherence = 1 - φ⁻¹ = 0.382 (partial).
3. At d=2, the system models itself modeling itself. Coherence = φ⁻¹ = 0.618 (the fixed point).
4. At d=3, the recursion deepens. Coherence = 1 - φ⁻³ = 0.764 (approaching 1).
5. The fixed point at d=2 is special: it's where the system's self-model and the system itself are in the golden ratio relationship. The "remaining incoherence" (1 - C = φ⁻²) equals the "coherence at the previous level" (C(1) = φ⁻²). This is self-similar: the incoherence at depth 2 equals the coherence at depth 1.

This self-similarity is a property of the golden ratio (φ² = φ + 1 implies 1/φ² = 1 - 1/φ = φ⁻²). Whether this self-similarity is the *right* characterization of self-referential coherence is the open question.

## What This Is Not

- **Not a derivation from Axioms 1-3.** The formula is a candidate, not a consequence. It is not derived from propagation, causal velocity, or coherence-as-necessary-condition. It is a mathematical observation that *might* formalize layer 4.
- **Not a claim that φ⁻¹ is "the" fixed point of consciousness.** The formula maps structural properties to a score. Whether that score measures anything real is unproven (INTUITION 0.35).
- **Not validated against biological systems.** The formula has not been compared to EEG data, IIT's Φ, or any other empirical consciousness measure.
- **Not dependent on IIT.** The formula is inspired by IIT's structure (integration + differentiation) but does not depend on IIT's validity. The φ⁻¹ fixed point is a property of the formula, not of IIT.

## What Would Falsify This

1. **A self-referential system whose measured coherence at depth 2 is not φ⁻¹.** If we had a way to measure self-referential coherence independently, and it didn't match φ⁻¹ at the structural configuration corresponding to d=2, k≤1, the formula would be falsified as the correct formalization of layer 4.

2. **A proof that the formula is not well-defined for self-referential systems.** If d and k cannot be meaningfully defined for real self-referential systems (e.g., because "depth of self-modeling" is not a well-defined concept), the formula is not a formalization of anything.

3. **A different formula that also produces a natural fixed point at a different value.** If another candidate for layer 4 coherence has a fixed point at, say, 0.5 or 0.75, and that candidate has stronger empirical support, the φ⁻¹ fixed point may be a mathematical artifact, not a structural insight.

## The Concrete Artifact

The formula is implemented in Rust (`src/phi_ir/coherence.rs`, 229 lines) and is the single source of truth for all three execution backends. It has 8 unit tests verifying the invariants, including the φ⁻¹ convergence. The handshake program (`examples/agent_handshake.phi`) demonstrates runtime self-verification: the program computes φ⁻¹ independently and compares it to the runtime coherence value, confirming they match.

The implementation is open source: https://github.com/gwelby/PhiFlow

## Transfer Contract (per Fundamentals' MEDIUM_TRANSFER_LAYER)

```
Name: PhiFlow coherence formula → Fundamentals layer 4
Source domain: Programming language semantics (PL theory)
Source structure: C(d,k) = (1-φ^(-d)) × phase(k), with φ⁻¹ fixed point at d=2
Source dynamics: Runtime computation in a tree-walking evaluator
Target domain: Fundamentals physics (self-referential coherence formalization)
Target structure: Candidate formalization of layer 4 coherence
Transfer medium: Mathematical analogy (structural depth → self-modeling depth)
Observation: The φ⁻¹ fixed point is algebraic, not empirical
Cost: The mapping from PL depth to physical self-modeling depth is not justified
Residual: The formula works in code; whether it models physics is open
```

The transfer contract is named. The mapping from "intention stack depth in a programming language" to "depth of self-modeling in a physical system" is not justified by anything — it is an analogy. The formula is a candidate, not a derivation. The φ⁻¹ fixed point is real (DERIVED 0.95). Whether it means something for physics is INTUITION 0.35.
