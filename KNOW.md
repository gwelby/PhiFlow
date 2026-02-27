# KNOW.md

Last updated: 2026-02-26

This file is the reality index for PhiFlow.  
Canonical explanation + use cases: [`PhiFlow_KNOW.md`](./PhiFlow_KNOW.md)

## Truth Priority
1. Executable verification results (`cargo build`, `cargo test`, integration sweeps)
2. Branch/worktree state (`git log`, `git status`, ahead/behind)
3. QSOP state/pattern logs
4. Narrative docs

If docs and executable behavior disagree, executable behavior wins.

## Current Verified Reality

### Workspace split
- `D:\Projects\PhiFlow` (`master`): docs/witness lane, local codebase currently older than compiler lane for runtime features.
- `D:\Projects\PhiFlow-compiler` (`compiler`): more advanced runtime pipeline lane (PhiIR/evaluator/VM/WASM work).
- `D:\Projects\PhiFlow-cleanup` (`cleanup`): still near initial commit from master viewpoint.
- `D:\Projects\PhiFlow-lang` (`language`): still near initial commit from master viewpoint.

### Branch status snapshot (2026-02-26)
- `master` is ahead/behind `origin/master` (diverged).
- `compiler` is ahead of `origin/compiler`.
- `cleanup` and `language` show minimal delta from initial commit.

### Build/test status
- In `D:\Projects\PhiFlow\PhiFlow` (master crate):
  - `cargo build --release` passes with warnings.
  - `cargo test` fails in `tests/performance_tests.rs` shots-scaling assertion due to simulator shot handling.
- In `D:\Projects\PhiFlow-compiler\PhiFlow`:
  - `cargo build --release` passes.
  - `cargo test --quiet` passes.
  - corpus sweep test passes (reports non-fatal dialect drift diagnostics).

## Known Drift Areas
1. Local master runtime path and parser behavior are not fully aligned with compiler-lane stream/PhiIR-era behavior.
2. Documentation across worktrees is not fully synchronized.
3. Historical entropy artifacts exist in master tree shape (duplicate-style paths and mixed-era tests).

## What To Do Next
1. Define canonical runtime lane (`compiler`) until merge reconciliation is complete.
2. Commit/stabilize compiler worktree pending files.
3. Merge/sync runtime features into master in controlled steps with tests.
4. Fix master simulator shot-count behavior and re-green `cargo test`.
5. Add CI gates that match local verification commands.
6. Continue keeping QSOP files factual and date-stamped.

## Related Files
- [`PhiFlow_KNOW.md`](./PhiFlow_KNOW.md): best plain-language explanation + uses
- [`D_PROJECTS_INTEGRATION_PLAYBOOK.md`](./D_PROJECTS_INTEGRATION_PLAYBOOK.md): concrete integration tracks across `D:\Projects`
- [`README.md`](./README.md): outer repo orientation and worktree strategy
- [`VISION.md`](./VISION.md): long-form why/architecture intent
- [`PhiFlow/QSOP/STATE.md`](./PhiFlow/QSOP/STATE.md): verified state ledger
- [`PhiFlow/QSOP/PATTERNS.md`](./PhiFlow/QSOP/PATTERNS.md): recurring bug/success patterns
- [`PhiFlow/QSOP/CHANGELOG.md`](./PhiFlow/QSOP/CHANGELOG.md): dated operational history
