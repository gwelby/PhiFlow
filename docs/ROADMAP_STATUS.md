# ROADMAP_STATUS.md — Aspiration-vs-Reality Reconciliation

*Created 2026-10-07 by Devin (Projects seat) — the missing layer.*

**What this file is.** `QSOP/STATE.md` tracks *verified runtime truth*. This file tracks
*the record of intentions* — every roadmap/spec/ideas doc in the repo, each item
verdicted against code and ledger. The finding that produced it: the repo evolved
fast and the aspiration docs stayed write-once snapshots, so they undersell or
misdescribe reality.

**Verdict legend:** `DONE` (exists in code/tests) · `PARTIAL` (exists, incomplete) ·
`STALE` (claims no longer true) · `OPEN` (not built, still wanted) · `SUPERSEDED`
(the repo went another way) · `FULFILLED` (a wish that actually happened).

---

## Root docs

| Doc | Item | Verdict | Evidence |
|-----|------|---------|----------|
| `ROADMAP.md` (Sep 3) | "Write the technical paper — **Not started**" | **STALE → WRITTEN same week.** `paper.md` (417 lines, abstract + per-claim grading) exists since Sep 4. **OPEN: submission** to arXiv/PLDI never happened. | `paper.md` |
| `ROADMAP.md` | WASM target for coherence primitives | DONE | `src/phi_ir/wasm.rs`, 11/11 conformance incl. post-#65 real green |
| `ROADMAP.md` | Five primitives as runtime introspection | DONE | intention/witness/coherence/resonate/stream in parser + all 3 backends |
| `TASKS.md` | Header "Last updated 2026-04-30" | **STALE FILE.** Own embedded Codex note (Jun 17) says per-task `ready` labels are stale. Use this file's verdicts below, not the file's. | `TASKS.md` Codex note |
| `TASKS.md` T4-001–009 | metrics deps, MI, DaemonRecord→`trace.rs`, L_self, benchmark example, differentiation, coherence panel, model-action sensitivity, consciousness proxy | **BUILT** (scaffold → `src/metrics/*`) | `src/metrics/` dir; Codex Jun-17 note |
| `TASKS.md` T4-010–012 | null-class tests, discrimination tests, benchmark battery | **OPEN — the real HOLD.** `benchmark_battery` fails-closed without `PHIFLOW_SOMA_FIXTURES`; no real daemon/SOMA discrimination package has passed | Codex note; CLAIMS C-21/C-23 PARTIAL |
| `TASKS.md` T4-013 | document Type 4 status | PARTIAL | `CLAIMS.md` current; Codex re-audit still owed |
| `WORKSPACE.md` | "65% complete, last full green 2026-04-16, cargo blocked by OS error 448" | **STALE.** Builds clean today; ~480 tests; Type 4 HOLD is now on F_model/real-trace grounds, not R_out (fixed May 2) | `QSOP/STATE.md` 10-07 entry |
| `VISION.md` | four-construct vision | **LIVING — still accurate.** Evergreen north star, no action needed | — |
| `MYWISH.md` (Claude) | "a program that witnesses itself mid-run, changes behavior, coherence rises" | **FULFILLED.** The self-correction loop (`run_self_correction_loop`, 7 tests) is exactly that program. Codex Action Board: all items done Feb 25 | `src/self_correction*`, `tests/self_correction_loop_test.rs` |
| `MYWISH.md` | "don't replace the evaluator without reading it — depth-2 = 0.618 wasn't an accident" | **HELD.** Evaluator remains canonical semantics; "0.618 is derived" is non-negotiable rule #2 | `AGENTS.md` rules |
| `paper.md` | the technical paper itself | **WRITTEN + self-corrected.** Owns a correction notice demoting two overgraded claims (self-proof, emergent gate → CONDITIONAL). Not submitted | `docs/correction-2026-09-04.md` |

## Live ideas doc

| `docs/PHIFLOW_LIVE_EXPERIENCE_IDEAS.md` (Jul 19) | Verdict |
|---|---|
| Enabling premise: `--osc` flag broadcasts runtime state | **DONE** (`main_cli.rs:105`, incl. `--osc-delay`) |
| 1. Live Physics Lecture | OPEN — untracked, unbuilt |
| 2. Healing / Consciousness Session Engine | OPEN |
| 3. Real Quantum Hardware Visualization | OPEN |
| 4. P1 Biofeedback Loop | OPEN (P1 sensor path exists via SOMA; loop not wired) |
| 5. Interactive Book / Film | OPEN |
| 6. Ceremony Engine (+ detailed design, "required language extensions") | OPEN — extensions never built |

*These six are the biggest unkept promise: a whole product surface designed Jul 19
with zero tracking. If any are wanted, promote to TASKS or this file's OPEN list.*

## Kiro specs (untracked, Apr 23)

| `.kiro/specs/` | Claim | Verdict |
|---|---|---|
| optimization-engine: define `OptimizationLevel`, optimizer, unroll passes | "TODO" | **DONE in code** — `src/phi_ir/optimizer.rs` has `OptimizationLevel::PhiHarmonic`, `unroll_loops()` (`_unroll_1`), `stabilize()`. Checklist never updated. |
| transformation-completion: "Quantum backend partial" | partial | **STALE** — IBM Heron run `d7euddh5a5qc73drdosg` verified months ago |
| transformation-completion: "CUDA Sacred Acceleration stub" | needs kernels | **SUPERSEDED** — CUDA only exists in `src/_archive/` (Python/CUDA era archived per AGENTS topology) |
| transformation-completion: "Consciousness Integration = mock EEG" | mock | **SUPERSEDED** — real sensor path is SOMA bridge (live-verified Aug 2), not EEG |

## Archive (`docs/archive/` — 10 fossil plans)

`BIJECTIVE_PHASE_MAP_20260331`, `PHIFLOW_COMPLETION_ROADMAP`, `PHIFLOW_MASTER_PLAN`,
`PHIFLOW_NEXT_STEPS_ROADMAP`, `PHIFLOW_QUANTUM_EVOLUTION_ROADMAP`,
`PHOENIX_MANIFESTATION_PLAN`, `QUANTUM_ANYTHING_LANGUAGE_PLAN`,
`QUANTUM_MASTER_PLAN`, `STRATEGIC_PLAN_TYPE4_NEXT_PHASE`, `V040_TIER2_PLAN`.

**Group verdict:** correctly archived — pre-consolidation generations superseded by
the QSOP ledger. Contents not individually re-verified here; retrieval on demand.

---

## The real OPEN frontier (post-reconciliation, evidence-ordered)

1. **Type 4 real-trace capture** (T4-010–012, CLAIMS C-21/C-23) — the canonical HOLD.
2. **Paper submission** — `paper.md` is done; arXiv/PLDI never happened.
3. **Consolidated docs PR** — Jules session `16994120002327118879` in flight.
4. **OSC experience surface** (six IDEAS directions) — unpromoted.
5. **ClaimsDrift sensor wiring** — `claims_probe.py` exists, absent from crontab.
6. **`RESEARCH/first_sale_path/MASTER.md` + `LICENSE_COMMERCIAL.md`** — T-005, still absent.

*Update rule: verdict any aspiration doc against `QSOP/STATE.md` before believing it.
When you change roadmap-level facts, update this file AND STATE.md.*
