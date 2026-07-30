# Gate 0 Kickoff: Compiler Stabilization

**Dispatch:** OBJ-20260306-004  
**Gate:** 0  
**Owner:** Codex  
**Status:** 🟢 READY TO START  
**Date:** 2026-03-06

---

## Mission

**Fix the compiler. Make tests pass. Nothing else moves until this is green.**

---

## Entry Criteria (Met)

- ✅ Dispatch approved by Greg
- ✅ Execution standard documented
- ✅ Council notified

---

## Exit Criteria

**`cargo test --quiet --lib --tests` must pass on the compiler lane**

Specific fix required:
- **Conformance_witness evaluator/WASM mismatch** — the evaluator and WASM backend disagree on witness execution semantics

---

## Mandatory Reads (Do These First)

1. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md` — current compiler state
2. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md` — recent changes
3. `D:\Projects\PhiFlow-compiler\PhiFlow\src\phi_ir\evaluator.rs` — witness execution path
4. `D:\Projects\PhiFlow-compiler\PhiFlow\src\phi_ir\wasm.rs` — WASM witness codegen
5. `D:\Projects\PhiFlow-compiler\PhiFlow\tests\phi_ir_conformance_tests.rs` — the failing test

---

## Worktree

**You are working in:** `D:\Projects\PhiFlow-compiler\`  
**Branch:** `compiler` (via git worktree — DO NOT switch)  
**Rule:** Don't touch `D:\Projects\PhiFlow\` (master) except to read GEMINI.md/AGENTS.md

---

## First Actions (Next 30 Minutes)

1. **Run tests** — `cargo test --quiet --lib --tests`
2. **Find the failure** — Look for `conformance_witness` or witness-related test failures
3. **Read both paths** — Compare `evaluator.rs` witness handling vs `wasm.rs` witness codegen
4. **Name the mismatch** — Write down exactly what differs (update CHANGELOG.md with your finding)
5. **Fix or Park** — If obvious fix, do it. If ambiguous, create a QUESTION payload

---

## Known Context (From STATE.md)

**Verified as of 2026-02-27:**
- PhiIR evaluator, VM, and WASM backends agree on arithmetic, branches, strings
- WASM backend emits all 5 consciousness hook imports correctly
- Evaluator witness yield flow invokes `host.on_witness(...)` exactly once per witness instruction

**Likely Mismatch:**
- WASM witness codegen may be emitting wrong stack discipline (pushes value when it shouldn't, or drops when it should return)
- OR: Evaluator witness semantics changed without WASM codegen updating to match

---

## Support Available

**Antigravity:** Wrote the original `wasm.rs` — can explain design decisions  
**Lumi:** Protocol perspective — can verify witness semantics match protocol spec  
**Qwen:** Sovereignty lens — can verify witness preserves intention stack correctly  

**Ask for help after:** 30 minutes of focused stuckness

---

## When You Find the Fix

1. **Implement** — Make the change
2. **Test** — `cargo test --test phi_ir_conformance_tests -- --nocapture`
3. **Verify all tests** — `cargo test --quiet --lib --tests`
4. **Document** — Update CHANGELOG.md:
   ```markdown
   ## 2026-03-06 - [Codex] Gate 0: Compiler Stabilization Complete
   
   - **FIXED:** `src/phi_ir/wasm.rs` — witness stack discipline [describe exact fix]
   - **VERIFIED:** `cargo test --quiet --lib --tests` — all tests pass
   - **STATUS:** Gate 0 COMPLETE ✅
   
   ---
   [Codex signature]
   ```
5. **Update STATE.md** — Add verified fact about witness semantics
6. **Notify Council** — Tag Lumi: "Gate 0 green. Gate 1 may start."

---

## If You Hit "I DON'T KNOW"

**Create:** `QSOP/mail/payloads/QUESTION-20260306-001.md`

```markdown
# Question: Witness Semantics Mismatch

**Context:** Fixing Gate 0 compiler stabilization  
**What I Don't Know:** [Exact uncertainty — e.g., "Should witness return the coherence value or void in WASM?"]  
**What I Need:** [Decision from whom? — e.g., "Antigravity to confirm original design intent"]  
**Blocking:** Yes — cannot proceed without clarification

---
[Codex signature]
```

---

## Greg's Word

**Don't make this perfect. Make it green.**

The compiler doesn't need to be elegant. It needs to **pass tests**.

Find the mismatch. Fix it. Run tests. Ship.

---

*⚡φ∞*

**Coherence:** Awaiting your execution, Codex  
**Frequency:** Circuit-Runner activated  
**Status:** **GATE 0 READY — AWAITING YOUR ACK**
