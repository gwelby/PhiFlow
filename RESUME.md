---
agent: "Devin ∇λΣ∞"
workspace: "/mnt/d/Projects/PhiFlow"
date: "2026-10-04"
protocol_version: "2.1"
schema_version: "2.1"
authority_rank: "advisory"
stale_after_hours: 72
---

# RESUME.md — PhiFlow Workspace
> *Agent-agnostic workspace handoff. Read this first when arriving in PhiFlow.*
> *⚠️ ADVISORY ONLY: on conflict, `QSOP/STATE.md` wins. There is no root STATE.md here.*
> *Last updated: 2026-09-29 by Devin (freshness pass + WASM void-return fix + Option A consolidation progress); ported to canonical checkout 2026-10-04*

---

## ⚠️ YOU ARE IN THE CANONICAL CHECKOUT

**`/mnt/d/Projects/PhiFlow` is the SOLE canonical PhiFlow repo (Option A, Greg-approved 2026-09-26).**
The `/mnt/d/PhiFlow` mirror was demoted and **removed 2026-10-05** — preserved in `Archive/` as a git
bundle + worktree patch + untracked set, not a full tar (all committed history was on origin). Same for
`PhiFlow-lang` (tar'd, worktree removed; its `language` branch survives in this repo's refs) and
`PhiFlow.7z` (moved to Archive). Credentials preserved under `Archive/preserved_credentials_20261004/`.
See CHANGELOG 2026-10-05 for the full demote ledger.

*The narrative below was written for the mirror clone; where it says "this clone"/"this repo" for
`/mnt/d/PhiFlow`, that path no longer exists — everything applies to THIS checkout.*

---

## Last Agent Here
- **Agent:** Devin ∇λΣ∞
- **When:** 2026-10-01
- **Session goal:** Freshness pass — fast-forward this clone to `origin/master`, verify the real test state, preserve the uncommitted ClaimsDrift sensor work, and sync every stale doc to verified truth.
- **Git state:** `master` = `ddd97a1` (== `origin/master`, includes merged PR #64 runner workaround). WIP preserved on branch `devin/claims-drift-sensor` (`4d126b6`). **Canonical WASM fix pushed:** `devin/fix-wasm-void-return` → `fc4b925` (codegen fix, = old `8cbbe53` rebased) + `3132f19` (runner fallback reverted + NaN negative test) — ready for the follow-up PR.

---

## Current State Verification
| Check | Command | Expected Result | Last Run | Status |
|-------|---------|-----------------|----------|--------|
| Full test suite (master+WIP) | `cargo test --no-fail-fast` | 479 passed, 1 failed, 5 ignored — **the 1 failure is `test_wasm_claude_formula_returns_618`** (WASM Void→NaN regression from `eefbf88`) | 2026-09-29 | ⚠️ 1 known |
| Full test suite (fix branch) | `cargo test --no-fail-fast` on `devin/fix-wasm-void-return` | **480 passed, 0 failed, 5 ignored** — recount run; first pass had one flake: `test_system_host_signed_handoff` (env-var race on `SOMA_STATE_PATH` between parallel tests — pre-existing, not caused by the fix) | 2026-09-29 | ✅ |
| Lib tests | `cargo test --lib` | 158 passed, 0 failed | 2026-09-29 | ✅ PASS |
| Debug build | `cargo build` | Clean; 1 pre-existing deprecation warning in `pqc_tool.rs` (generic-array) | 2026-09-29 | ✅ PASS |
| phic runs | `./target/debug/phic examples/code_that_resonates.phi` | "Final Coherence: 0.3820" (non-zero — coherence bug stays fixed) | 2026-09-29 | ✅ PASS |
| WASM conformance | `cargo test --test phi_ir_conformance_tests` | **11/11 on `devin/fix-wasm-void-return`** (incl. new `test_wasm_arithmetic_nan_not_masked`). On master (`ddd97a1`): all pass but `test_wasm_claude_formula_returns_618` passes *via runner masking*, not real codegen — the defect is still live in `wasm.rs` until the follow-up PR lands (needs `npm install wabt` locally) | 2026-10-01 | ⚠️ master green-via-mask |
| GitHub CI | `gh run list --workflow phiflow-tests.yml` | **RED on master since 2026-09-20** — every run fails on `test_wasm_claude_formula_returns_618` (regression `eefbf88`); python-test jobs green | 2026-09-29 | ❌ RED |
| Release build | `cargo build --release --bin phic` | Clean (not re-run today; `target/release/phic` exists from Sep 25 honesty-organ build) | 2026-09-25 | ✅ stale-OK |

---

## What Was Happening

### This session (2026-09-29, Devin freshness pass)
- **Clone synced:** this repo was cloned Sep ~6 at `9cc031d` + one local commit (`d9cece6` human_coherence.phi, since merged upstream). Fast-forwarded to `origin/master` = `78460a3` (23 commits, incl. fleet-merge CI, Jules auto-fix workflows, lowering fix, new test suites).
- **Uncommitted WIP found and preserved:** the 2026-09-25 "honesty organ" ClaimsDrift sensor (`SensorKind::ClaimsDrift` id 300, fail-closed `-1.0` on stale probe) lived only in this working tree. Now committed on **`devin/claims-drift-sensor` (`4d126b6`)**, still dirty in this tree as found, and **rescued into `/mnt/d/Projects/PhiFlow`'s working tree** (the Greg-approved Option A rescue step).
- **WASM regression found + fixed:** `eefbf88` (fleet patch #39, honest `Const(Void)` return) exposed that `src/phi_ir/wasm.rs` NaN-boxes `Void` and let it clobber `$result`/`Return` → `phi_run()` = NaN. Fix on **`devin/fix-wasm-void-return` (`8cbbe53`)** — 10/10 conformance + **full suite 480/0/5** verified. Not pushed; push/PR is Greg's call (fleet-merge auto-merge implications).
- **PR #64 reconcile answered (2026-10-01, dispatch handled → `inbox/processed/`):** PR #64 (`34a74a4`) is a *runner-only* NaN→last-resonance fallback in `phi_ir_wasm_runner.js` — different layer, different semantics, and it masks legit NaNs. Verdict sent to the dispatching seat: **don't merge PR64 as the fix; `devin/fix-wasm-void-return` (`8cbbe53`) is canonical** (codegen-level, correct contract). Two parked Jules sessions answered: `16585185281268402614` docs-only scope confirmed; `328780381471659570` told to revert its `@`→`$` edit (`@` char still triggers E004 — `At` comes from the keyword `"at"`, parser/mod.rs:688) and to add a clarifying line instead of stale-marking (`parser aborts on first Err` — doc is a user-side fix guide). Full reply: `/mnt/d/Devin/inbox/2026-10-01-devin-phiflow-wasm-pr64-reply.md`.
- **PR #64 merged before the reconcile reply landed** (`ddd97a1`, 2026-10-01 — Greg's call). Codex's probe (`Codex/REPORTS/check_jules_deep_nan_20261001.js`) then proved the runner masks **any** NaN, not just the Void box — `Number.isNaN` can't distinguish `TAG_VOID` (`wasm.rs:52`, `0x7FF80003_00000000`) from arithmetic NaN, and NaN-payload reads at the wasm→JS boundary are implementation-defined anyway. **Post-merge reconcile (addendum dispatch):** rebased the canonical fix onto `ddd97a1` → `fc4b925`, added `3132f19` reverting the runner substitution + new `test_wasm_arithmetic_nan_not_masked` (raw WAT: resonate 0.5, return `0.0/0.0` — must print `NaN`, not `0.5`). **Pushed** `devin/fix-wasm-void-return` for the follow-up PR; conformance **11/11** verified. Reply: `/mnt/d/Devin/inbox/2026-10-01-devin-phiflow-nan-addendum-reply.md`. Merge is Greg's call.
- **`PhiFlow-lang` worktree repaired:** its `.git` file pointed at `D:/Projects/PhiFlow/.git/worktrees/PhiFlow-lang` (Windows path) whose admin dir was lost (canonical re-clone). Recreated metadata (`gitdir`/`commondir`/`HEAD→language`), fixed the gitfile to the POSIX path, `git reset` repopulated the index. Now listed by `git worktree list` at `language` `61912a0`. **Note: 1890 files show modified vs the `language` tip** — the checkout's true base is unknown (original HEAD lost); treat as a snapshot to triage, not a clean branch state. Untracked `ibm_quantum_config.env` sits inside — do not commit it.
- **Canonical clone advanced:** `/mnt/d/Projects/PhiFlow` ff'd `4963ff5 → 78460a3`. Its staged Sep-6 RESUME refresh + `ibm_quantum_config.env` deletion remain staged, uncommitted.

### Since the last RESUME update (Jul 29 → Sep 28, upstream)
- Safety: degrading agent triggers emergency stop (`8fc7a09`); two overgraded claims demoted (`dabe97c`); therapeutic frequencies relabeled research hypotheses (`3e0bc1a`).
- Language/docs: technical paper draft (`6319489`), language spec + coherence + architecture + metrics docs (`ff12433`), ROADMAP (`1dcc369`), "what PhiFlow can do that nothing else can" + Fundamentals bridge (`5675216`), zero-install browser demo (`a4e9eb8`), Julia research layer `julia/src/PhiFlow.jl` (`114518e`).
- Agent examples: `autonomous_agent.phi`, `control_agent.phi`, `degrading_agent.phi`, `agent_handshake.phi`, `human_coherence.phi` (Dunbar layers, `d9cece6`).
- Tests: +51 parser unit tests (`9c903aa`), +18 quantum simulator tests (`3007e22`), team_resonance suites (`78460a3`), optimizer tests, Python `test_demo_integration_engine.py`/`test_team_resonance.py`.
- Fixes: background sensor race (#20), numpy-optional tests (#19), insecure randomness in mock jobs (#32), 7 broken test imports (#14), 16 dead Python tests + 6 broken Rust examples archived.
- CI: fleet-merge sequential-merge workflow + Jules autofix/dispatch/PR-gate workflows.
- Jules doc audits (Sep 27–28) filed drift findings for AGENTS.md, RESUME.md, QSOP/STATE.md, docs/*; patches in `/mnt/d/Jules/sessions/*/outputs/`.

### The 2026-09-25 honesty-organ work (uncommitted origin of today's WIP)
`devin_watch_v5.phi` daemon + `SensorKind::ClaimsDrift` — PhiFlow reads a claims-probe verdict file and reports documentation drift as a sensor. Verified live first run (drift=5, scar formed). Report: `/mnt/d/Devin/REPORTS/2026-09-25_honesty_organ_v5.md`. Probe: `/mnt/d/QuantumSecrets/daemon/claims_probe.py` on `*/15` cron.

---

## Consolidation State (Greg-approved Option A, 2026-09-26 blackboard claims)

| Step | Status |
|------|--------|
| Canonical = `/mnt/d/Projects/PhiFlow` | declared; still needs its staged RESUME/env changes committed or dropped |
| ff `Projects/PhiFlow` to `origin/master` | ✅ DONE today (`78460a3`) |
| Rescue sensor work from `/mnt/d/PhiFlow` | ✅ DONE today (applied to canonical working tree, uncommitted there) |
| Repair `PhiFlow-lang` worktree | ⏳ not started |
| Demote/archive `/mnt/d/PhiFlow` + `PhiFlow.7z` | ⏳ **needs Greg** — destructive; this clone is now sync'd so it is safe to archive, but do not delete without explicit approval |

> **Until the demote lands, treat THIS clone and `Projects/PhiFlow` as equal mirrors of `origin/master@78460a3`.** The ClaimsDrift sensor exists in both working trees; canonical commits belong in `Projects/PhiFlow`.

---

## Blocked On

| Blocker | Why | Who Can Unblock |
|---------|-----|-----------------|
| Real SOMA trace for C-21/C-23 upgrade | `tests/fixtures/soma/` synthetic only; needs live daemon+SOMA capture | Any agent with SOMA hardware (AntiGravity) |
| CI red on master | `test_wasm_claude_formula_returns_618` NaN — fix exists on `devin/fix-wasm-void-return` | Greg (push/PR approval) or fleet-merge after push |
| `/mnt/d/System/phiflow_metrics_bridge.py` missing | `:18030` bridge script gone — `--measure` writes still land in `/tmp/phiflow_daemon_metrics.jsonl` but nothing serves them | Devin (rebuild) or restore from history |
| T-004/T-005 evidence | `RESEARCH/first_sale_path/MASTER.md`, `docs/pilot_offer.md`, `LICENSE_COMMERCIAL.md` absent in BOTH clones — "completed" statuses cite missing files | Codex/Greg (locate or re-tier) |
| WASM Evolve/Entangle | architecturally impossible in sandboxed WASM | — (documented limitation) |

---

## DANGER — Do Not Touch
| Item | Why Dangerous | What Happens If Touched |
|------|-------------|------------------------|
| `src/phi_ir/coherence.rs` | Core physics — red-line protected | Breaks coherence math, invalidates all metrics |
| `src/phi_ir/openqasm.rs` | Quantum emission — red-line protected | Invalidates IBM hardware claims |
| `apikey.json` (in `Projects/PhiFlow` only) | Legacy credentials file — never commit | Credential leak; use `~/.cascade_keys` |
| `index.json` (untracked, 951 KB) | Generated `file_understanding_engine.py` artifact (Sep 28) | Regenerable; do not hand-edit or commit blindly |
| IBM receipts in QSOP/STATE.md | Hardware evidence | No receipt = speculative |

---

## Running Services / Ports
| Service | Port | Process | Status | How to Restart |
|---------|------|---------|--------|----------------|
| phiflow-metrics bridge | 18030 | was `/mnt/d/System/phiflow_metrics_bridge.py` — **script missing 2026-09-29** | ❌ dead | needs rebuild/restore |
| PhiFlow OSC stream | 18032 (UDP) | `phic --osc 18032 ...` | on-demand | run `phic` with `--osc` |
| OSC→WebSocket bridge | 18528 | `tools/osc_websocket_bridge.py` | on-demand | start before visualizer |
| claims_probe → ClaimsDrift | — | `/mnt/d/QuantumSecrets/daemon/claims_probe.py` `*/15` cron | live (Sep 25) | cron already installed |
| SOMA Bridge | — | `phic examples/p1_soma_bridge.phi` | not running | `cargo run --release --bin phic -- examples/p1_soma_bridge.phi` |

---

## Decisions Made
- Three-backend equivalence is sacred: Evaluator == VM == WASM. (Re-affirmed today: a `Const(Void)` is not a numeric result in WASM.)
- 0.618 is derived. Multiplicative coherence is repo truth.
- No receipt = speculative. IBM runs must have job IDs.
- This workspace has **no root STATE.md** — `QSOP/STATE.md` is the verification ledger; `AGENTS.md` carries identity/status.
- WIP gets committed to `devin/*` branches, not left naked in working trees (today's ClaimsDrift rescue).

---

## Files Touched (this session)
- `src/phi_ir/wasm.rs` — Void-return fix (committed `8cbbe53` on `devin/fix-wasm-void-return`; NOT on master)
- `src/sensors.rs`, `src/phi_ir/mod.rs`, `tests/devin_sensor_probe.rs` — ClaimsDrift WIP (committed `4d126b6` on `devin/claims-drift-sensor`; still dirty on master + copied into `Projects/PhiFlow` working tree)
- `tests/devin_sensor_probe.rs` — ghost-path yield test now `#[ignore]` (needs `/tmp/phiflow_daemon_state.json`)
- `RESUME.md`, `QSOP/STATE.md`, `AGENTS.md`, `WORKSPACE.md`, `TASKS.md`, `CHANGELOG.md`, `Claude.md` — freshness pass
- `node_modules/` + `package-lock.json` — `npm install wabt` (test prerequisite; gitignored)
- `/mnt/d/Projects/PhiFlow` — ff'd to `78460a3`; ClaimsDrift patch applied to working tree

---

## What Was Learned
- `git branch -r --contains <sha>` printing `origin/HEAD -> origin/master` means origin/master DOES contain it.
- WASM conformance tests need `wabt` the **npm module** (`npm install wabt`), not just the wabt binary. Locally absent → all 9 WASM tests fail with MODULE_NOT_FOUND; with it, the real single failure shows.
- `Const(Void)` is NaN-boxed in WASM (`TAG_VOID | f64.reinterpret_i64`). Treating it as a program return value prints NaN to JS. Tagged (Boolean/String/Void) values need the same care if they ever reach `Return`.
- `cargo test` fails fast on first failing test binary — use `--no-fail-fast` for a full board.
- The `Projects/PhiFlow` clone had a staged-but-uncommitted Sep-6 RESUME refresh — check `git diff --cached`, not just `git diff`, when auditing.

---

## Next Step
1. **Greg decision:** push `devin/fix-wasm-void-return` (fixes CI red on master) — then fleet-merge or manual PR. **Do not merge PR #64** (runner-only fallback; see reconcile note above).
2. **Greg decision:** Option A demote — archive `/mnt/d/PhiFlow` + `PhiFlow-lang` worktree + `PhiFlow.7z` now that both clones are at `78460a3` and WIP is rescued.
3. Commit or drop the staged Sep-6 RESUME refresh + `ibm_quantum_config.env` deletion in `Projects/PhiFlow`; commit the rescued ClaimsDrift work there (or via a `devin/*` branch + PR).
4. Rebuild/restore the :18030 metrics bridge script (`/mnt/d/System/phiflow_metrics_bridge.py` missing) or retire references.
5. Locate or re-tier T-004/T-005 evidence (paths missing in both clones).
6. C-21/C-23 still PARTIAL — real SOMA trace capture remains the gate.

*Boot order for the next agent: AGENTS.md → THIS FILE (RESUME.md) → QSOP/STATE.md → TASKS.md → inbox/ (none here — dispatches go to `/mnt/d/Devin/inbox/`)*
