# CHANGELOG

## 2026-02-26 - [Codex] Human-first docs sweep + D:\Projects implementation mapping
- Scope: root documentation and QSOP state witness alignment in `D:\Projects\PhiFlow` (master worktree).
- Updated human-first docs:
  - `README.md`
  - `PhiFlow_KNOW.md` (new canonical plain-language explainer)
  - `KNOW.md` (reduced to factual reality index + truth hierarchy)
  - `VISION.md` (rewritten with current branch reality and 90-day execution plan)
- Updated witness ledger:
  - `PhiFlow/QSOP/STATE.md` with dated `2026-02-26` section for human-first sync and verified `D:\Projects` integration candidates.
- Verified local `D:\Projects` candidates by path inspection:
  - `UniversalProcessor`, `ResonanceMatrix`, `MCP`, `P1_Companion`, `QDrive`, `Quantum-Fonts`.
- Design decision:
  - Keep `compiler` lane as runtime truth until merge reconciliation; keep master lane as documentation/witness authority with explicit branch-state qualifiers.

## 2026-02-25 - [Codex] Master truth sync from cross-worktree witness
- Scope: documentation/QSOP lane in `D:\Projects\PhiFlow` (master worktree only).
- Read and cross-checked:
  - `VISION.md`
  - `PhiFlow/QSOP/STATE.md`
  - `PhiFlow/QSOP/PATTERNS.md`
  - `git log compiler --oneline -10`
  - `git log cleanup --oneline -10`
  - `git log language --oneline -10`
  - `git show compiler:PhiFlow/QSOP/{STATE.md,PATTERNS.md,CHANGELOG.md}`
- Updated `PhiFlow/QSOP/STATE.md`:
  - Added a dated `2026-02-25` cross-worktree witness section separating verified branch-log facts from probable items.
  - Recorded current branch reality from master view: compiler active; cleanup/language currently unchanged since initial commit.
  - Added runtime-trend note that compiler treats `phi_ir::evaluator` as canonical semantics path.
- Updated `PhiFlow/QSOP/PATTERNS.md`:
  - Added `P-3` (WASM stream loop-back signal loss).
  - Added `P-4` (cross-agent cargo lock contention).
- Design decision:
  - Keep master QSOP as a strict witness ledger based on observable evidence (`git log`/`git show`) and mark anything not directly executed in this worktree as probable until locally re-verified.
- Verification:
  - `python PhiFlow/QSOP/tools/validate_packets.py` -> `Validation passed: 3 objective packet(s), 9 ack packet(s), 0 issues.`
  - `python PhiFlow/QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48` -> all steps passed; metrics/audit refreshed in `PhiFlow/QSOP/metrics/`.

## 2026-02-21 - [Codex] Weaver seed/contract/loom schema draft
- Added Weaver trio:
  - `PhiFlow/QSOP/weaver/the_weaver.seed.yaml`
  - `PhiFlow/QSOP/weaver/the_weaver.contract.yaml`
  - `PhiFlow/QSOP/weaver/the_loom_ledger_schema.yaml`
- Added operational flow doc:
  - `PhiFlow/QSOP/weaver/README.md`
- Added dispatch-ready activation objective:
  - `PhiFlow/QSOP/mail/payloads/OBJ-20260221-003.md`
  - `PhiFlow/QSOP/mail/objectives/OBJ-20260221-003.json` (initial dispatch packet)
- Design decision:
  - Keep filesystem path `QSOP` for compatibility.
  - Use "The Loom" as conceptual alias in docs/contracts until migration tooling is ready.
- Included first dispatch-ready flow for Team of Teams:
  - Codex draft/update -> Antigravity resonance review -> objective dispatch -> ritual verification.
- Verification run:
  - `python PhiFlow/QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48` -> all steps passed.

## 2026-02-21 - [Codex] OBJ-20260221-003 completed with reference-grounded staging synthesis
- Verified references were provided in compiler worktree and imported into this workspace:
  - `PhiFlow/QSOP/weaver/references/cascade_reference.yaml`
  - `PhiFlow/QSOP/weaver/references/tesla_reference.yaml`
- Created review-only staging trio:
  - `PhiFlow/QSOP/weaver/staging/the_weaver.seed.yaml`
  - `PhiFlow/QSOP/weaver/staging/the_weaver.contract.yaml`
  - `PhiFlow/QSOP/weaver/staging/the_loom_ledger_schema.yaml`
  - `PhiFlow/QSOP/weaver/staging/README.md`
- Closed objective flow:
  - Updated `PhiFlow/QSOP/mail/objectives/OBJ-20260221-003.json` to `completed`
  - Added `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-003-codex.json`
- Verification run:
  - `python PhiFlow/QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48` -> pass
  - Metrics now: objectives=3, acked=3, completed=3, reopened=1

## 2026-02-21 - [Codex] Added blocked/recovered + reopened objective scenario
- Added payload:
  - `PhiFlow/QSOP/mail/payloads/OBJ-20260221-002.md`
- Added objective:
  - `PhiFlow/QSOP/mail/objectives/OBJ-20260221-002.json`
- Added ACK transition chain:
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-01.json` (accepted)
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-02.json` (blocked)
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-03.json` (in_progress)
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-04.json` (completed)
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-05.json` (blocked; reopened)
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-06.json` (in_progress)
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-002-codex-07.json` (completed)
- Verification run:
  - `python PhiFlow/QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48` -> all steps passed.
- Metrics impact:
  - totals: objectives=2, acked=2, completed=2
  - reopen tracking now exercised: `reopened_objectives=1`, `reopen_rate=0.5`
  - validation/audit remain clean (0 issues, 0 warnings).

## 2026-02-21 - [Codex] Seeded first live objective cycle
- Added payload:
  - `PhiFlow/QSOP/mail/payloads/OBJ-20260221-001.md`
- Added objective packet:
  - `PhiFlow/QSOP/mail/objectives/OBJ-20260221-001.json`
- Added completion ack:
  - `PhiFlow/QSOP/mail/acks/ACK-OBJ-20260221-001-codex.json`
- Used `python PhiFlow/QSOP/tools/compute_payload_checksum.py QSOP/mail/payloads/OBJ-20260221-001.md` to bind checksum.
- Verification run:
  - `python PhiFlow/QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48` -> all steps passed.
- Metrics moved from zero-history to live:
  - total=1, acked=1, completed=1, verification_coverage=1.0, reopen_rate=0.0.

## 2026-02-21 - [Codex] Phase 2 tooling upgrades (checksum + SLA + one-command runner)
- Added checksum enforcement for objective payloads:
  - `PhiFlow/QSOP/tools/qsop_packet_lib.py` (`verify_objective_payload_checksum`)
  - `PhiFlow/QSOP/tools/validate_packets.py` now validates `checksum` vs `payload_path` content.
- Added SLA-aware weekly audit:
  - `PhiFlow/QSOP/tools/weekly_qsop_audit.py` now checks:
    - objective age without ack (`--pending-ack-sla-hours`, default 24h)
    - stale `in_progress` ack age (`--in-progress-sla-hours`, default 48h)
- Added one-command ritual runner:
  - `PhiFlow/QSOP/tools/run_all.py` executes validate -> metrics -> audit with SLA args.
- Added checksum helper:
  - `PhiFlow/QSOP/tools/compute_payload_checksum.py`
- Updated docs:
  - `PhiFlow/QSOP/tools/README.md`
  - `PhiFlow/QSOP/metrics/README.md`
- Verification run:
  - `python PhiFlow/QSOP/tools/validate_packets.py` -> pass
  - `python PhiFlow/QSOP/tools/log_objective_metrics.py` -> pass, metrics written
  - `python PhiFlow/QSOP/tools/weekly_qsop_audit.py` -> pass, audit written
  - `python PhiFlow/QSOP/tools/run_all.py --pending-ack-sla-hours 24 --in-progress-sla-hours 48` -> all steps passed

## 2026-02-21 - [Codex] Week 1 instrumentation shipped
- Added protocol scaffolding:
  - `PhiFlow/QSOP/mail/objectives/.gitkeep`
  - `PhiFlow/QSOP/mail/acks/.gitkeep`
  - `PhiFlow/QSOP/mail/payloads/.gitkeep`
  - `PhiFlow/QSOP/mail/templates/objective.template.json`
  - `PhiFlow/QSOP/mail/templates/ack.template.json`
- Added schema references:
  - `PhiFlow/QSOP/schemas/objective.schema.json`
  - `PhiFlow/QSOP/schemas/ack.schema.json`
- Added tooling:
  - `PhiFlow/QSOP/tools/qsop_packet_lib.py`
  - `PhiFlow/QSOP/tools/validate_packets.py`
  - `PhiFlow/QSOP/tools/log_objective_metrics.py`
  - `PhiFlow/QSOP/tools/weekly_qsop_audit.py`
  - `PhiFlow/QSOP/tools/README.md`
- Ran tooling successfully:
  - `python PhiFlow/QSOP/tools/validate_packets.py` -> pass (0 issues)
  - `python PhiFlow/QSOP/tools/log_objective_metrics.py` -> wrote `QSOP/metrics/objective_metrics.json`
  - `python PhiFlow/QSOP/tools/weekly_qsop_audit.py` -> wrote `QSOP/metrics/weekly_audit_20260221.md`
- Design decision: metrics tolerate empty history and invalid packets (reported, not fatal) so adoption can start immediately without blocking.

## 2026-02-21 - [Codex] Self-evolution plan added
- Added `PhiFlow/QSOP/CODEX_SELF_EVOLUTION_PLAN.md`.
- Captured Codex-specific growth tracks: execution quality, memory quality, coordination quality, research cadence.
- Added explicit `Me Time` distill rhythm (post-objective, daily refinement, weekly deep block).
- Defined measurable metrics and 30-day evolution sequence to reduce fuzz and increase verified throughput.

## 2026-02-21 - [Codex] Codex Operating System hardening
- Added `PhiFlow/AGENTS.md` as inner-repo agent contract (startup order, operating contract, handoff format).
- Added `PhiFlow/codex.toml` with explicit rules chain.
- Added rule set:
  - `PhiFlow/.codex/rules/00-core.md`
  - `PhiFlow/.codex/rules/10-qsop-loop.md`
  - `PhiFlow/.codex/rules/20-team-sync.md`
- Added `PhiFlow/QSOP/CODEX_EXCELLENCE_PLAYBOOK.md` with per-session + weekly improvement loops.
- Decision: codify protocol and memory as files first, then evolve skills/multi-agent automation on top.

## 2026-02-10 - Initial QSOP bootstrap
- ADDED to STATE: Full project architecture (parser, interpreter, CLI, constructs)
- ADDED to STATE: What exists vs what doesn't (no WASM/quantum/hardware backends yet)
- ADDED to STATE: Build and run commands
- NEW PATTERN: P-1 keyword-as-variable collision
- NEW PATTERN: P-2 newline sensitivity in statement parsing
- NEW PATTERN: S-1 four constructs map to QSOP operations
- NEW PATTERN: S-2 sacred frequency detection with tolerance band
- QSOP bootstrapped from session where all four constructs were designed and implemented
