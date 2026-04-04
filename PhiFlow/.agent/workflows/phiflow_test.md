# /phiflow_test

## Purpose
Executes the full Rust test suite for the PhiFlow compiler and updates the QSOP STATE.md with the results.

## Steps
1. **Locate Compiler:** Change directory to `D:\Projects\PhiFlow-compiler\PhiFlow\`.
2. **Execute Tests:** Run `cargo test --all-targets -- --nocapture`.
3. **Analyze Results:** Capture the number of passed/failed tests.
4. **Update State:** 
   - Open `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md`.
   - Update the "Verified" section with the current timestamp and test summary (e.g., "12/12 tests passed").
5. **Report:** Respond with a concise summary of the run.
