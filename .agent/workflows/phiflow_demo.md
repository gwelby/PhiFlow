# /phiflow_demo

## Purpose
Runs the canonical PhiFlow demo and verifies the golden outputs (Value + Coherence).

## Steps
1. **Locate Compiler:** Change directory to `D:\Projects\PhiFlow-compiler\PhiFlow\`.
2. **Run Demo:** Run `cargo run --example phiflow_demo`.
3. **Verify Output:** 
   - Check for `Number(84.0)`.
   - Check for coherence score `0.6180`.
4. **Log Action:** Add an entry to `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md` attributed to the current agent.
5. **Respond:** Confirm whether the demo matches the golden baseline.
