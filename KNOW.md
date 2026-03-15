# KNOW.md

Last updated: 2026-03-06

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
- `D:\Projects\PhiFlow` (`master`): docs/witness lane and, as of 2026-03-06, the most stable directly runnable crate surface.
- `D:\Projects\PhiFlow-compiler` (`compiler`): more advanced runtime pipeline lane (PhiIR/evaluator/VM/WASM/MCP), but currently in active repair with a dirty worktree.
- `D:\Projects\PhiFlow-cleanup` (`cleanup`): clean worktree with minimal unique commits; available capacity for structural cleanup.
- `D:\Projects\PhiFlow-lang` (`language`): clean worktree with minimal unique commits; available capacity for language work.

### Branch status snapshot (2026-03-06)
- `master...compiler` = 11 / 12 unique commits.
- `master...cleanup` = 23 / 1 unique commits.
- `master...language` = 23 / 1 unique commits.
- `master` and `compiler` both have dirty working trees; `cleanup` and `language` are clean.

### Build/test status
- In `D:\Projects\PhiFlow\PhiFlow` (master crate):
  - `cargo test --quiet` passed on 2026-03-06 with warnings only.
- In `D:\Projects\PhiFlow-compiler\PhiFlow`:
  - `cargo test --quiet --lib --tests` reaches the main test suite but currently fails in `tests/phi_ir_conformance_tests.rs::conformance_witness` due to evaluator/WASM mismatch (`lhs=0`, `rhs=NaN`).
  - Full `cargo test --quiet` currently fails earlier while compiling multiple examples that import `phiflow`, and also reports missing `rlib` forms for several dependencies.
  - Existing compiler-lane docs claiming full green status are therefore stale and need to be treated as historical, not current truth.

## Known Drift Areas
1. `master` is now greener than the docs say, while `compiler` is less green than the docs say; truth has inverted since the 2026-02-26 snapshot.
2. Runtime contracts across evaluator/VM/WASM/browser-host surfaces are not fully aligned yet, as shown by the live `conformance_witness` mismatch.
3. Documentation across worktrees is not fully synchronized.
4. Historical entropy artifacts still exist in the outer repository shape and cleanup branch has not executed the structural reduction plan yet.

## What To Do Next
1. Restore `compiler` end-to-end green status, starting with `conformance_witness` and example-target compilation.
2. Commit/stabilize compiler worktree pending files once the verification surface is green again.
3. Keep `master` as the stable demo/docs lane until compiler repairs are complete.
4. Use `cleanup` and `language` as active capacity lanes instead of leaving them near-idle.
5. Add CI gates that distinguish core tests from example/build surfaces so breakage is visible earlier.
6. Continue keeping QSOP files factual and date-stamped.

## Related Files
- [`PhiFlow_KNOW.md`](./PhiFlow_KNOW.md): best plain-language explanation + uses
- [`D_PROJECTS_INTEGRATION_PLAYBOOK.md`](./D_PROJECTS_INTEGRATION_PLAYBOOK.md): concrete integration tracks across `D:\Projects`
- [`README.md`](./README.md): outer repo orientation and worktree strategy
- [`VISION.md`](./VISION.md): long-form why/architecture intent
- [`PhiFlow/QSOP/STATE.md`](./PhiFlow/QSOP/STATE.md): verified state ledger
- [`PhiFlow/QSOP/PATTERNS.md`](./PhiFlow/QSOP/PATTERNS.md): recurring bug/success patterns
- [`PhiFlow/QSOP/CHANGELOG.md`](./PhiFlow/QSOP/CHANGELOG.md): dated operational history
