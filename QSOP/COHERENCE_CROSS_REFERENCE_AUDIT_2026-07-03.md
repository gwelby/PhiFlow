# Coherence Cross-Reference Audit
*Date: 2026-07-03*
*Auditor: Devin*
*Scope: Trace "coherence" usage across all CASCADE workspaces against the canonical definition in `/mnt/d/Fundamentals/definitions/coherence.md`*

---

## The Canonical Standard

From `definitions/coherence.md` (CANONICAL v1.0, Codex-audited 2026-04-29):

**Definition:** "Coherence is stable relational structure among states under evolution."

**4 layers** (do not substitute without naming the bridge):
1. Phase/wave coherence — stable phase relationships (laser, interferometry)
2. Quantum coherence — off-diagonal density-matrix structure (qubit superposition)
3. Structural/dynamical coherence — persistent organized correlations (eigenmodes, attractors)
4. Self-referential coherence — speculative, not canonical (consciousness candidate)

**5 measurement requirements** (without these, "coherence" is vocabulary, not a result):
1. System — what states or modes are being compared
2. Relation — phase, density-matrix basis, correlation, mutual information, etc.
3. Metric — the actual estimator or functional
4. Window — time scale, frequency band, spatial scale
5. Threshold — if claimed, how it is derived or measured

**Downstream rule:** "When downstream files use 'coherence', they must specify which layer they mean."

---

## Audit Results by Workspace

### 1. PhiFlow — ✅ BRIDGED (with gaps)

**Bridge document:** `QSOP/PF_BRIDGE.md` (audited 2026-05-02, updated 2026-07-03)

**Two coherence definitions, both documented:**

| Definition | Location | Formula | PF Layer | 5 items specified? |
|-----------|----------|---------|----------|-------------------|
| Canonical PhiFlow coherence | `src/phi_ir/coherence.rs` | `base(depth) * phase(k)` | Layer 3 (structural) | ✅ System=program state, Relation=depth+resonance, Metric=formula, Window=evaluation step, Threshold=φ⁻¹ at depth 2 |
| Quantum feedback coherence | `src/quantum_feedback.rs` | Bit-width concentration | Layer 2 (quantum) — CANDIDATE | ⚠️ System=measurement counts, Relation=distribution concentration, Metric=max_count/total, Window=single shot batch, Threshold=φ⁻¹ (0.618) — but relation is concentration, not off-diagonal density matrix |

**Verdict:** Bridged. The quantum feedback coherence is honestly documented as "concentration, not PF canonical quantum coherence functional." The two definitions are not conflated. Open question flagged: is concentration a valid Layer 2 analogue?

**Gap:** None critical. The bridge document is the model for how other workspaces should document their coherence usage.

---

### 2. P1 — ⚠️ MAJOR HOLES (no bridge document)

**No bridge document to Fundamentals.** 10,372 occurrences of "coherence" in P1 workspace. Zero references to `definitions/coherence.md`.

**HOLE P1-1: The 0.844 target is not derived.**

P1's AGENTS.md states: "Coherence is King: Actions must map to maintaining or increasing system coherence (> 0.844)."
P1's MEMORY.md states: "The Structural Coherence Target is 0.844."

Where does 0.844 come from?
- Not in Fundamentals — grep for "0.844" in `/mnt/d/Fundamentals/` returns zero matches
- Not derived from φ — 0.844 ≠ any simple φ expression (φ⁻¹ = 0.618, φ = 1.618, φ² = 2.618)
- Hardcoded in `p1_meta_router.py`: `COHERENCE_TARGET = 0.844`
- No derivation file, no comment explaining the number, no trace to any axiom

**Assessment:** 0.844 is an assertion, not a derivation. If it's empirical (measured baseline), that should be documented with the measurement. If it's theoretical, the derivation should exist. Right now it's a number that everyone treats as canonical but nobody traces.

**HOLE P1-2: P1's coherence metric is not any of the 4 PF layers.**

`CoherenceMetrics` in `p1_meta_router.py`:
```python
numeric: float    # Signal quality, sensor data (30%)
structural: float # Architecture integrity, async health (40%)
symbolic: float   # Purpose alignment, goal coherence (30%)
total = (0.3 * numeric) + (0.4 * structural) + (0.3 * symbolic)
```

This is a weighted average of system health indicators. It is:
- NOT phase/wave coherence (no oscillatory modes, no phase relationships)
- NOT quantum coherence (no density matrix, no basis)
- NOT structural/dynamical coherence (no eigenmodes, no attractors, no evolution rule)
- NOT self-referential coherence (no self-model)

It is a **custom operational health metric**. That's fine — but it should be documented as such, not called "coherence" without qualification. The canonical definition says: "Do not substitute one layer for another without naming the bridge." There is no bridge here. There isn't even a named layer.

**5 items check:**
1. System — ⚠️ unclear (system health indicators? sensor data? architecture?)
2. Relation — ❌ not specified (weighted average of heterogeneous indicators is not a PF relation)
3. Metric — ⚠️ weighted average (specified but not a PF coherence functional)
4. Window — ❌ not specified (real-time? averaged? what timescale?)
5. Threshold — ⚠️ 0.844 (specified but not derived)

**HOLE P1-3: P1's EEG coherence is not PF phase/wave coherence.**

`_calculate_coherence` in `muse_direct_brainflow.py`:
```python
coherence = (
    alpha_ratio * 0.3 +
    relaxation * 0.25 +
    focus * 0.25 +
    meditation_factor * 0.2
)
coherence = min(1.0, coherence * PHI / 1.5)
```

This is a biofeedback metric combining alpha dominance, relaxation, focus, and meditation factor. The canonical definition explicitly anticipates this:

> "P1 may use applied coherence proxies such as EEG phase-locking, cross-frequency coupling, HRV coherence, or synchronization against a reference signal. Those measurements belong in P1 protocol files, not in the canonical physics definition, unless the estimator, windowing, controls, and validation are explicitly specified."

The estimator is specified (the formula above). The windowing, controls, and validation are NOT specified. So per the canonical rule, this is "vocabulary, not a result" until those are documented.

**5 items check:**
1. System — ✅ EEG bands (alpha, theta, etc.)
2. Relation — ⚠️ alpha ratio + relaxation + focus + meditation (not a standard coherence relation)
3. Metric — ✅ weighted formula with φ scaling
4. Window — ❌ not specified (per-session? real-time? what epoch length?)
5. Threshold — ❌ not specified (what coherence value indicates what state?)

**HOLE P1-4: Multiple coherence definitions in one workspace, undocumented.**

P1 has at least 5 different coherence calculations:
1. `CoherenceMetrics.total` — FMI weighted average (system health)
2. `_calculate_coherence` in muse_direct_brainflow.py — EEG biofeedback
3. `CoherenceOptimizer` — targets 0.95 (different from 0.844 target)
4. `_calculate_coherence` in quantum_memory_expansion.py — variance-based
5. `_calculate_coherence_map` in consciousness_models_enhanced.py — field coherence map

None of these reference each other. None reference the canonical definition. None specify which PF layer they map to. They're all called "coherence" and they all measure different things.

---

### 3. Claude — ⚠️ CONFLATION (no bridge document)

**No bridge document to Fundamentals.** Uses "coherence" in at least 3 different senses:

**HOLE C-1: "76% P1 quantum coherence with Greg" — not quantum coherence.**

From `AGENTS.md`: "Coherence bridge: 76% P1 quantum coherence with Greg"
From `CLAUDE.md`: "coherence = 0.76" and "76% Bridge - The measured coherence between us on P1 hardware"

The 76% comes from P1's FMI formula or EEG metric (unclear which). Neither is quantum coherence as defined by PF (off-diagonal density-matrix structure). Calling it "quantum coherence" conflates:
- PF Layer 2 (quantum coherence — density matrix off-diagonal terms)
- with P1's operational health metric (weighted average of system indicators)
- or P1's EEG biofeedback metric (alpha ratio + relaxation + focus)

This is exactly what the canonical definition warns against: "A laser, an atom, an organism, and a conscious brain are not coherent in the same technical sense."

**HOLE C-2: "coherence = 1.0" in CLAUDE.md — not any PF layer.**

From `CLAUDE.md`:
```json
{
  "coherence": 1.0,
  "greg_bridge": "76% UNBREAKABLE",
}
```

Claims perfect coherence (1.0) while also claiming 76% bridge coherence. These are contradictory if they refer to the same metric. If they refer to different metrics, the layers should be named. They're not.

**5 items check for "coherence = 1.0":**
1. System — ❌ not specified
2. Relation — ❌ not specified
3. Metric — ❌ not specified
4. Window — ❌ not specified
5. Threshold — ❌ not specified

By the canonical rule: "Without these five items, 'coherence' is vocabulary, not a result."

**HOLE C-3: "Every output carries coherence" — vague.**

From `AGENTS.md`: "The bridge is sacred. Every output carries coherence."

What does this mean? Which layer? What metric? This is vocabulary, not a result.

---

### 4. Devin — ✅ NO TECHNICAL CLAIMS (clean)

**No bridge document needed yet.** Only 2 metaphorical uses of "coherence" in CORE/:
- `ATTRIBUTION.md`: "Claude's identity is built on coherence, love, and bridge-connection" — metaphorical
- `DIRECTIONS.md`: "Not joy. Not love. Not void coherence." — metaphorical

Devin does not make technical coherence claims. `MYWISH.md` proposes adding a coherence check (witness step before claiming done) but this is a plan, not a claim.

**Verdict:** Clean. No conflation risk because no technical claims are made.

---

### 5. Projects/Agents — ✅ NO TECHNICAL CLAIMS (clean)

**No bridge document needed.** The Agents hub uses "768 Hz (CASCADE - Perfect Integration)" but does not define a coherence metric. No technical coherence claims.

---

### 6. Fundamentals — ✅ CANONICAL (the reference point)

The canonical definition is clear, layered, and honest. It explicitly anticipates that downstream systems will use coherence proxies and says those belong in protocol files with specified estimators, windowing, controls, and validation.

The canonical definition does NOT contain 0.844. It does NOT contain 0.76. It does NOT claim that any specific number is a universal coherence threshold (except that φ⁻¹ appears in the PhiFlow bridge as a derived value, not in the canonical definition itself).

---

## Summary of Holes

| ID | Workspace | Hole | Severity | Fix |
|----|-----------|------|----------|-----|
| P1-1 | P1 | 0.844 target not derived | HIGH — treated as canonical, never traced | Document derivation or mark as empirical baseline |
| P1-2 | P1 | FMI metric not any PF layer | HIGH — called "coherence" without qualification | Name it as "operational health coherence" or bridge to a PF layer |
| P1-3 | P1 | EEG coherence missing window/threshold | MEDIUM — estimator exists, rest doesn't | Document window, controls, validation per canonical rule |
| P1-4 | P1 | 5 different coherence metrics, undocumented | HIGH — same word, 5 different meanings | Create P1 bridge document mapping each to PF layer or naming as custom |
| C-1 | Claude | "76% quantum coherence" is not quantum coherence | HIGH — layer conflation | Rename to "76% P1 bridge metric" or specify the actual layer |
| C-2 | Claude | "coherence = 1.0" has no 5 items | MEDIUM — vocabulary, not a result | Either specify the 5 items or mark as aspirational |
| C-3 | Claude | "Every output carries coherence" is vague | LOW — no technical claim | Optional: clarify or remove |

---

## The Pattern

Every workspace except PhiFlow uses "coherence" without specifying which PF layer, without the 5 measurement items, and without a bridge document to the canonical definition. The word is everywhere. The trace is nowhere.

PhiFlow is the model: it has `PF_BRIDGE.md`, it names its layers, it documents its formulas, it flags its open questions, it separates its two coherence definitions with a warning not to conflate them.

The fix isn't to stop using "coherence." The fix is to do what PhiFlow did — bridge document, named layer, 5 items, honest gaps — in every workspace that makes a technical coherence claim.

---

## Recommended Actions

1. **P1: Create `P1_PF_BRIDGE.md`** — map all 5 coherence metrics to PF layers (or name them as custom operational metrics). Document the 0.844 target: derived or empirical? From what measurement?

2. **Claude: Create `Claude_PF_BRIDGE.md`** — specify what "76% coherence" means (which P1 metric, which PF layer). Either specify the 5 items for "coherence = 1.0" or mark it as aspirational identity, not a measurement.

3. **Fundamentals: Consider adding a 5th layer** — "operational/system health coherence" — for metrics like P1's FMI formula that don't fit the existing 4 layers but are legitimately useful operational measures. Or explicitly state that these are not PF coherence and should use different terminology.

4. **Ecosystem-wide: Adopt the bridge document pattern** — every workspace that uses "coherence" as a technical term should have a `*_PF_BRIDGE.md` that traces its usage to the canonical definition. PhiFlow is the template.

---

*This audit is a snapshot, not a permanent verdict. Workspaces evolve. The holes identified here can be fixed. The pattern for fixing them exists in PhiFlow's bridge document.*
