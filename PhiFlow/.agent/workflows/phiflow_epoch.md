# /phiflow_epoch

## Purpose
The master workflow for transitioning the compiler to a new development phase or feature set.

## Steps
1. **Ingest State:** Read `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md`.
2. **Verify Main:** Execute `/phiflow_test` to ensure the current state is stable.
3. **Plan Feature:** Create a new plan document in `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\mail\payloads\`.
4. **Initialize Epoch:**
   - Update `STATE.md` with the new "Current Epoch" title.
   - Append a "START" entry to `CHANGELOG.md`.
5. **Scaffold:** Create necessary new files or stubs in `src/`.
6. **Report:** Provide the new epoch roadmap to the user.
