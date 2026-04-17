# Bijective Phase Map Implementation
**Date:** 2026-03-31
**Agent:** Qwen Code
**Status:** ✅ Implementation Complete - Verification Blocked by Host Memory

---

## Summary

Implemented the Bijective Phase Map coherence formula in both the VM (`src/phi_ir/vm.rs`) and Evaluator (`src/phi_ir/evaluator.rs`) as specified in AGENTS.md Task #1.

---

## Changes Made

### 1. `src/phi_ir/vm.rs` - `compute_coherence()`

**Before:**
```rust
fn compute_coherence(&self) -> f64 {
    let depth = self.intention_stack.len();
    let resonance_count: usize = self.resonance_field.values().map(|v| v.len()).sum();

    if depth == 0 && resonance_count == 0 {
        return 0.0;
    }

    let intention_coherence = if depth > 0 {
        1.0 - PHI.powi(-(depth as i32))
    } else {
        0.0
    };
    let resonance_bonus = (resonance_count as f64 * 0.05).min(0.2);
    (intention_coherence + resonance_bonus).min(1.0)
}
```

**After:**
```rust
fn compute_coherence(&self) -> f64 {
    let depth = self.intention_stack.len();
    
    // Bijective Phase Map: k is the winding number (resonance cardinality)
    // k=1 → perfect coherence (1.0), k>1 → logarithmic decay
    // Find the maximum resonance cardinality across all intentions
    let max_k: usize = self.resonance_field.values()
        .map(|v| v.len())
        .max()
        .unwrap_or(0);

    // If no resonances but has intentions, k=1 (primitive winding)
    let k = if depth > 0 && max_k == 0 {
        1
    } else if max_k == 0 {
        return 0.0;
    } else {
        max_k
    };

    // Bijective phase map formula:
    // k=1 → 1.0 (perfect coherence for primitive winding)
    // k>1 → 1.0 - ln(k) / ln(2π) (logarithmic decay for multi-winding)
    if k == 1 {
        1.0
    } else {
        let decay = (k as f64).ln() / std::f64::consts::TAU.ln();
        (1.0 - decay).max(0.0)
    }
}
```

### 2. `src/phi_ir/evaluator.rs` - `compute_coherence()`

Applied the same formula to maintain three-backend equivalence.

### 3. Test Updates

**`tests/phi_ir_conformance_tests.rs`:**
- Updated `conformance_resonate_then_coherence` to expect `1.0` (k=1) instead of `φ⁻¹ + 0.05`

**`src/phi_ir/vm.rs` tests:**
- Updated `vm_coherence_tracks_intention_and_resonance` to expect `~1.0` (k=1 bijective)
- Added `vm_coherence_bijective_k2_decay` test for k=2 decay formula

**`tests/phi_ir_evaluator_tests.rs`:**
- Updated `test_resonance_adds_bonus_to_coherence` to expect `1.0` (k=1)
- Added `test_bijective_k2_decay` test for k=2 decay formula

---

## Formula Derivation

The Bijective Phase Map uses winding number k (resonance cardinality):

| k | Formula | Result | Meaning |
|---|---------|--------|---------|
| 0 | N/A | 0.0 | No intentions, no resonance |
| 1 | 1.0 | 1.0 | Perfect coherence (primitive winding) |
| 2 | 1.0 - ln(2)/ln(2π) | ~0.623 | First decay level |
| 3 | 1.0 - ln(3)/ln(2π) | ~0.403 | Second decay level |
| k | 1.0 - ln(k)/ln(2π) | decays | Logarithmic decay |

**Key insight:** k represents the maximum number of resonances in any single intention scope. A single resonance (k=1) achieves perfect coherence. Multiple resonances in the same intention (k>1) represent contradiction and decay coherence logarithmically.

---

## Verification Status

### ✅ Compilation
- `cargo check --lib` passes successfully

### ⚠️ Tests Blocked
- Windows host memory pressure prevents `cargo test` execution
- Observed errors: `os error 1455` (paging file too small), `0xc000012d` (stack overflow)
- This is a known issue documented in QSOP/STATE.md (2026-03-15)

### Recommended Next Steps
1. Run tests on a machine with more RAM or reduced memory pressure
2. Verify three-backend equivalence: `cargo test --lib openqasm` + conformance tests
3. Update QSOP/STATE.md with dated verification entry

---

## Files Modified

| File | Lines Changed | Purpose |
|------|---------------|---------|
| `src/phi_ir/vm.rs` | ~40 | Bijective Phase Map implementation + tests |
| `src/phi_ir/evaluator.rs` | ~40 | Bijective Phase Map implementation |
| `tests/phi_ir_conformance_tests.rs` | ~5 | Updated expected values |
| `tests/phi_ir_evaluator_tests.rs` | ~60 | Updated tests + new k=2 test |

---

## Truth Order Compliance

Per AGENTS.md:
- ✅ Code changes made to both VM and Evaluator
- ✅ Tests updated to reflect new formula
- ⚠️ Verification pending due to host constraints
- ⏭️ QSOP/STATE.md update pending test verification

---

## Notes for Next Agent

When tests can run:
1. Run `cargo test --quiet phi_ir_vm_tests phi_ir_evaluator_tests phi_ir_conformance_tests`
2. Verify all coherence-related tests pass
3. Update QSOP/STATE.md with:
   ```markdown
   ## Verified (2026-03-31) [Bijective Phase Map Implementation]
   - Bijective Phase Map implemented in vm.rs + evaluator.rs
   - k=1 → 1.0, k>1 → 1.0 - ln(k)/ln(2π)
   - Tests updated: conformance_resonate_then_coherence, vm_coherence_*, test_bijective_k2_decay
   - Three-backend equivalence: [CONFIRMED/FAILED]
   ```
