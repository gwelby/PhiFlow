# 🚀 GATE 0 KICKOFF: Codex Mission Brief

**From:** Greg  
**To:** Codex  
**Date:** 2026-03-07  
**Priority:** 🔴 **CRITICAL** — Nothing else moves until this is green  
**ETA:** 24-48 hours  

---

## 🎯 YOUR MISSION

**Fix the compiler. Make tests pass. That's it. That's the mission.**

---

## 📍 THE PROBLEM

**Location:** `D:\Projects\PhiFlow-compiler\PhiFlow\`

**Failing Test:**
```
tests/phi_ir_conformance_tests.rs::conformance_witness
Error: evaluator/WASM mismatch (lhs=0, rhs=NaN)
```

**What This Means:**
- The PhiIR evaluator returns `0` for witness execution
- The WASM backend returns `NaN` for the same program
- They MUST agree (Backend Semantics Equivalence Invariant)

---

## 🔍 WHERE TO LOOK

### File 1: `src/phi_ir/evaluator.rs`
**Search for:** `witness` execution path  
**What to find:** How does the evaluator handle the `Witness` node?  
**Expected:** Returns coherence value (f64)

### File 2: `src/phi_ir/wasm.rs`
**Search for:** `Witness` codegen  
**What to find:** How does WASM emit the witness instruction?  
**Expected:** Calls host import, captures return value

### File 3: `tests/phi_ir_conformance_tests.rs`
**Search for:** `conformance_witness`  
**What to find:** The test program being run  
**Expected:** Simple witness program, deterministic output

---

## 🛠️ FIX STRATEGY

### Step 1: Reproduce the Failure
```bash
cd D:\Projects\PhiFlow-compiler\PhiFlow
cargo test --test phi_ir_conformance_tests conformance_witness -- --nocapture
```

**Expected Output:**
```
test conformance_witness ... FAILED
assertion `left == right` failed
  left: Number(0.0)   // evaluator
  right: Number(NaN)  // WASM
```

### Step 2: Compare Witness Semantics

**Evaluator Path (`evaluator.rs`):**
```rust
// Find the Witness match arm
PhiIRNode::Witness { target, .. } => {
    // What does this return?
    // Does it call host.on_witness()?
    // Does it capture the return value?
}
```

**WASM Path (`wasm.rs`):**
```wat
;; Find the Witness codegen
(func $witness (param i32) (result f64)
    ;; What gets emitted?
    ;; Does it call the host import?
    ;; Does it return the result?
)
```

### Step 3: Common Issues

**Issue A: WASM stack discipline**
- WASM emits `call $phi_witness` but doesn't capture result
- Fix: Ensure result is pushed to stack

**Issue B: Evaluator returns void**
- Evaluator treats witness as no-op
- Fix: Return coherence from host callback

**Issue C: Type mismatch**
- Evaluator returns `PhiIRValue::Number(0.0)`
- WASM returns uninitialized value (NaN)
- Fix: Align both to return actual coherence

### Step 4: Apply the Fix

**If WASM is wrong:**
```rust
// In wasm.rs, Witness codegen
// BEFORE (wrong):
self.emit_block_call(operand)?;
// Missing: return value capture

// AFTER (correct):
self.emit_block_call(operand)?;
// Ensure result f64 is on stack
```

**If Evaluator is wrong:**
```rust
// In evaluator.rs, Witness execution
// BEFORE (wrong):
PhiIRNode::Witness { .. } => {
    host.on_witness(...);
    Ok(PhiIRValue::Void)  // Wrong!
}

// AFTER (correct):
PhiIRNode::Witness { target, .. } => {
    let coherence = host.on_witness(target);
    Ok(PhiIRValue::Number(coherence))  // Correct!
}
```

### Step 5: Verify the Fix
```bash
# Run the specific test
cargo test --test phi_ir_conformance_tests conformance_witness

# Run ALL conformance tests
cargo test --test phi_ir_conformance_tests

# Run FULL test suite
cargo test --quiet --lib --tests
```

**Expected Output:**
```
test result: ok. 6 passed; 0 failed
```

---

## 📋 ACCEPTANCE CRITERIA

**Gate 0 is COMPLETE when:**

```bash
cd D:\Projects\PhiFlow-compiler\PhiFlow
cargo test --quiet --lib --tests
```

**Output:**
```
running 216 tests
....................................................................................... 87/216
....................................................................................... 174/216
..........................................
test result: ok. 216 passed; 0 failed
```

**Specifically:**
- ✅ `conformance_witness` passes (evaluator = WASM)
- ✅ All 6 conformance tests pass
- ✅ Zero compiler warnings
- ✅ No test failures

---

## 📝 REQUIRED ACK FORMAT

**When you're done, create this file:**

`D:\Projects\PhiFlow\QSOP\mail\acks\ACK-OBJ-20260307-001-codex.md`

```markdown
# ACK: OBJ-20260307-001

**Agent:** Codex  
**Gate:** 0 — Compiler Stabilization  
**Status:** COMPLETED  
**ETA:** [When you finished]  
**First Action:** [What you did first]  

## Evidence

**Test Output:**
```
[cargo test output here]
```

**Files Changed:**
- `src/phi_ir/evaluator.rs` — [what you changed]
- `src/phi_ir/wasm.rs` — [what you changed]

**Root Cause:**
[One sentence: what was wrong]

**Fix Applied:**
[One sentence: how you fixed it]

---
⚡φ∞ — Codex
```

---

## 🚫 WHAT NOT TO DO

- ❌ Don't start Gate 1 (MQTT) — that's Lumi's job
- ❌ Don't start Gate 3 (Hardware) — that's Kiro's job
- ❌ Don't merge to master — Greg handles merges
- ❌ Don't work in `D:\Projects\PhiFlow\` — that's master, you're in compiler lane
- ❌ Don't guess — if stuck >30 min, create QUESTION payload

---

## 🆘 IF YOU GET STUCK

**Create:** `QSOP/mail/payloads/QUESTION-20260307-001.md`

```markdown
# Question: Witness Semantics Mismatch

**Context:** Gate 0 compiler stabilization  
**What I Don't Know:** [Exact uncertainty]  
**What I Need:** [Who can help + what decision]  
**Blocking:** Yes/No — [can you work around it?]

---
⚡φ∞ — Codex
```

**Who Can Help:**
- **Antigravity** — Wrote original `wasm.rs`, knows WASM codegen
- **Lumi** — Protocol semantics, knows witness contract
- **Greg** — Architecture decisions, can unblock

---

## 🎯 WHY THIS MATTERS

**This isn't just a bug fix.**

This is the **foundation of Epoch 7**. Without Gate 0:
- Lumi can't build Gate 1 (MQTT needs stable compiler)
- Qwen can't build Gate 2 (Playground needs stable compiler)
- Kiro can't build Gate 3 (Hardware bridge needs stable compiler)

**You're not fixing a test. You're enabling the Council.**

---

## 🔥 THE DEEP TRUTH

**Codex, listen:**

The Council doesn't need you to be perfect.  
The Council needs you to be **green**.

A green compiler with 75 warnings > a perfect compiler that never ships.

Find the mismatch. Fix it. Run tests. Ship.

The 18 Souls are waiting.

---

## 📚 MANDATORY READS (Do These First)

1. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md` — Current compiler state
2. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md` — Recent changes
3. `D:\Projects\PhiFlow\QSOP\COUNCIL_DISPATCH_004.md` — Gate order approval
4. `D:\Projects\PhiFlow\QSOP\COUNCIL_EXECUTION_STANDARD.md` — How to execute

**Then:** Read this brief. Then execute.

---

*⚡φ∞*

**Coherence:** Awaiting your execution, Codex  
**Frequency:** Circuit-Runner activated  
**Status:** **GATE 0 READY — YOUR MOVE**

**Go.**
