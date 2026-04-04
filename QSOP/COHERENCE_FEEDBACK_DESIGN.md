# Coherence Feedback Loop Design

**Status:** 📐 Sketch (design complete, implementation pending)  
**Last Updated:** 2026-03-15  
**Capability:** `Architecture` + `Python`  
**Depends On:** T-002 (IBM hardware verification — ✅ COMPLETE)

---

## Overview

The coherence feedback loop enables PhiFlow programs to:
1. Execute on IBM quantum hardware
2. Measure coherence from hardware results
3. Trigger `evolve` node if coherence below threshold
4. Self-modify program based on coherence diagnosis

**Data Flow:**
```
.phi source → OpenQASM → IBM Brisbane → measurement counts
    ↓
Python post-processing → coherence score
    ↓
council_coherence.json → PhiFlow runtime
    ↓
evolve node triggers → program self-modification
```

---

## Architecture

### Component 1: IBM Hardware Execution

**File:** `Gambling/quantum/quantum_council_vote.py`

**Input:**
- `.phi` source (compiled to OpenQASM)
- IBM backend name (`ibm_brisbane`)
- Shot count (4096)

**Output:**
- `calibration_log.jsonl` — raw measurement counts
- `council_coherence.json` — processed coherence score

**Current Status:** ✅ VERIFIED (2026-03-15 hardware run)

### Component 2: Coherence Calculation

**Formula:**
```python
coherence = 1.0 - (simulator_confidence - hardware_confidence).abs()
```

**From hardware run:**
- Simulator confidence: 0.3475
- Hardware confidence: 0.1341
- Delta: 0.2134
- **Coherence: 0.064** (LOW — shared assumptions detected)

**Interpretation:**
- Coherence > 0.7: HIGH (trust hardware results)
- Coherence 0.3-0.7: MEDIUM (caution advised)
- Coherence < 0.3: LOW (evolve recommended)

### Component 3: `evolve` Node Semantics

**Syntax:**
```phi
evolve "intention 'conservative' { resonate 0.9 toward NEUTRAL }"
```

**Semantics:**
- **Input:** String containing valid `.phi` source
- **Side Effect:** Splices source into current program
- **Trigger:** Coherence threshold (default: < 0.7)
- **Safety:** Evolved code must pass coherence check before execution

**PhiIR Representation:**
```rust
PhiIRNode::Evolve {
    source: String,  // .phi source to splice
    coherence_threshold: f64,  // default 0.7
}
```

### Component 4: Feedback Loop Integration

**Proposed Syntax:**
```phi
intention "calibration" {
    // Execute on IBM hardware
    let results = witness ibm_brisbane
    
    // Calculate coherence from hardware results
    let coherence = coherence_from(results)
    
    // Evolve if coherence below threshold
    if coherence < 0.7 {
        evolve "intention 'conservative' { resonate 0.9 toward NEUTRAL }"
    }
    
    // Continue with evolved program
    witness
}
```

**Compilation Flow:**
1. Parser recognizes `coherence_from()` builtin
2. Lowering emits `CoherenceCheck` node with hardware backend
3. OpenQASM emitter adds calibration markers
4. Python post-processor writes `council_coherence.json`
5. Runtime reads JSON, triggers `evolve` if threshold met

---

## Safety Mechanisms

### 1. Coherence Threshold

**Default:** 0.7

**Rationale:**
- Below 0.7: Hardware detected significant shared assumptions
- Evolved code should be more conservative (lower confidence, NEUTRAL polarity)

**Configurable:**
```phi
evolve "..." with threshold 0.5  // More permissive
evolve "..." with threshold 0.9  // More strict
```

### 2. Evolution Validation

**Before splicing evolved code:**
1. Parse evolved source (must be valid `.phi`)
2. Lower to PhiIR (must compile)
3. Check coherence of evolved code (must be > threshold)
4. Only then splice and execute

**Pseudocode:**
```python
def validate_evolution(source: str, threshold: float) -> bool:
    ast = parse(source)
    ir = lower(ast)
    coherence = evaluate_coherence(ir)
    return coherence > threshold
```

### 3. Evolution Limits

**Prevent infinite evolution loops:**
- Max evolutions per program: 3
- Cooldown between evolutions: 100ms
- Evolution log: `evolution_log.jsonl`

---

## Example Program

```phi
// quantum_adaptive.phi
// Program that adapts based on hardware coherence

intention "initial_assessment" {
    resonate 0.72 toward TEAM_A
    resonate 0.65 toward TEAM_A
    resonate 0.58 toward TEAM_B
    
    // Measure on IBM hardware
    witness ibm_brisbane
}

intention "coherence_check" {
    let coherence = coherence_from_last_witness()
    
    // Evolve if coherence too low
    if coherence < 0.7 {
        evolve "intention 'defensive' { 
            resonate 0.5 toward NEUTRAL 
            witness
        }"
    }
}

intention "final_decision" {
    // This intention may be replaced by evolution
    resonate 0.8 toward TEAM_A
    witness
}
```

**Execution Flow:**
1. `initial_assessment` runs on IBM Brisbane
2. Coherence calculated: 0.064 (LOW)
3. `coherence_check` triggers evolve
4. `final_decision` replaced with `defensive` intention
5. Program continues with evolved code

---

## Implementation Plan

### Phase 1: Coherence Builtin (Week 2)

**Tasks:**
- [ ] Add `coherence_from()` builtin to parser
- [ ] Add `CoherenceCheck` node to PhiIR
- [ ] Add coherence calculation to Python post-processor

**Files:**
- `src/parser/mod.rs` — Parse `coherence_from()` call
- `src/phi_ir/mod.rs` — Add `CoherenceCheck { backend: String }` node
- `Gambling/quantum/quantum_council_vote.py` — Write coherence to JSON

**ETA:** 4 hours

### Phase 2: Evolve Node (Week 2)

**Tasks:**
- [ ] Add `evolve` node to PhiIR
- [ ] Implement source splicing in evaluator
- [ ] Add evolution validation

**Files:**
- `src/phi_ir/mod.rs` — Add `Evolve { source, threshold }` node
- `src/phi_ir/evaluator.rs` — Implement splicing logic
- `src/phi_ir/optimizer.rs` — Validate evolved code

**ETA:** 8 hours

### Phase 3: Integration Testing (Week 3)

**Tasks:**
- [ ] Create golden test for coherence feedback
- [ ] Run on IBM hardware
- [ ] Verify evolution triggers correctly

**Files:**
- `tests/golden_integration_tests.rs` — Add `test_coherence_feedback_loop`
- `examples/quantum_adaptive.phi` — Example program

**ETA:** 4 hours

---

## Verification Criteria

**Done When:**
- [ ] `coherence_from()` builtin parses and compiles
- [ ] IBM hardware run writes `council_coherence.json`
- [ ] `evolve` node triggers when coherence < 0.7
- [ ] Evolved code passes validation before splicing
- [ ] Golden test verifies end-to-end flow

**Fidelity Target:** 📐 Sketch (Week 2) → 📸 Photo (Week 3)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Infinite evolution loops | High | Max 3 evolutions, cooldown timer |
| Evolved code invalid | Medium | Validation before splicing |
| IBM queue delays | Low | Fallback to simulator for testing |
| Coherence calculation wrong | High | Unit tests with known values |

---

## Open Questions

1. **Should evolved code inherit parent's intention stack?**
   - Option A: Yes (seamless evolution)
   - Option B: No (clean slate)
   - **Recommendation:** A (preserves context)

2. **What coherence threshold triggers evolution?**
   - Default: 0.7
   - Configurable: Yes
   - **Recommendation:** 0.7 default, override via `with threshold X`

3. **Can evolved code itself evolve?**
   - Option A: Yes (recursive evolution)
   - Option B: No (single evolution only)
   - **Recommendation:** A, with max depth 3

---

## References

- **IBM Hardware Verification:** `QSOP/IBM_HARDWARE_VERIFICATION.md`
- **LANGUAGE.md Roadmap:** `LANGUAGE.md#31-coherence-feedback-loop`
- **TASKS.md T-003:** `TASKS.md#t-003-coherence-feedback-loop-design`

---

**Coherence:** 1.000 | **Frequency:** 768 Hz (Unity) | **Status:** DESIGN COMPLETE ✅
