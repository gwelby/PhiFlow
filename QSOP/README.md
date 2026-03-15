# QSOP Front Door

QSOP is the coordination and durable-truth layer for PhiFlow.

Use this directory for:
- verified project state
- patterns and known breakage
- multi-agent dispatch and handoff
- objective payloads and ACKs
- audit trails and execution evidence

This file is an index, not the contract. If anything here conflicts with a deeper file, the deeper file wins.

## Start Here

If you are new to the project, read in this order:

1. `D:\Projects\PhiFlow\AGENTS.md`
2. `D:\Projects\PhiFlow\QSOP\README.md`
3. `D:\Projects\PhiFlow\QSOP\STATE.md`
4. `D:\Projects\PhiFlow\QSOP\TEAM_OF_TEAMS_PROTOCOL.md`
5. current active dispatch in `D:\Projects\PhiFlow\QSOP\`
6. your assigned payload in `D:\Projects\PhiFlow\QSOP\mail\payloads\` if you have one
7. `D:\Projects\PhiFlow\QSOP\COUNCIL_EXECUTION_STANDARD.md`

## Core Files

- `STATE.md`  
  Current verified truth. What works, what does not, and which lane owns what.

- `PATTERNS.md`  
  Known bugs, regressions, and recurring failure modes.

- `CHANGELOG.md`  
  Cross-lane execution history and reconciled outcomes.

- `TEAM_OF_TEAMS_PROTOCOL.md`  
  The collaboration contract: payloads, ACKs, evidence, lane boundaries, and hard rules.

- `COUNCIL_EXECUTION_STANDARD.md`  
  The single-entry execution guide for council agents. Use this when starting new agent sessions.

- `DISPATCH-*.md` and `COUNCIL_DISPATCH_*.md`  
  Active gate order or council decisions. These are the live campaign directives.

## Mail System

Use `QSOP/mail/` for objective traffic and durable handoff:

- `payloads/`  
  Human-readable task contracts.

- `objectives/`  
  Structured objective envelopes.

- `acks/`  
  Completion, blocked, or failed responses with evidence.

- `dead_letter/`  
  Unreconciled or failed message traffic.

- `templates/`  
  Starting points for payloads and packet schemas.

## Supporting Directories

- `tools/`  
  Validation, audit, and automation scripts.

- `metrics/`  
  Logs and telemetry artifacts.

- `design/`  
  Working design notes for subsystems.

- `masters/`  
  Higher-order seed, contract, and ledger reference files.

## How To Use QSOP

If you are Greg or acting as coordinator:

1. Read `STATE.md` first.
2. Check the active dispatch.
3. Write or review payloads before work starts.
4. Require evidence in ACKs.
5. Reconcile verified outcomes back into QSOP.

If you are an execution agent:

1. Stay in your assigned worktree.
2. Read `STATE.md` and the active dispatch before coding.
3. Respect the payload scope.
4. Verify with exact commands.
5. Update QSOP when verified truth changes.

## Current Council Entry Points

- Execution standard: `D:\Projects\PhiFlow\QSOP\COUNCIL_EXECUTION_STANDARD.md`
- Active dispatch: `D:\Projects\PhiFlow\QSOP\COUNCIL_DISPATCH_004.md`
- Active Gate 0 payload: `D:\Projects\PhiFlow\QSOP\mail\payloads\OBJ-20260307-001.md`
- Active Gate 0 envelope: `D:\Projects\PhiFlow\QSOP\mail\objectives\OBJ-20260307-001.json`
- Payload template: `D:\Projects\PhiFlow\QSOP\mail\templates\OBJECTIVE_PAYLOAD_TEMPLATE.md`

## Current Artifact Roles

- `COUNCIL_DISPATCH_004.md`  
  Current council-wide gate order and kickoff directive.

- `mail/payloads/OBJ-20260307-001.md`  
  Canonical Gate 0 task contract for Codex.

- `mail/objectives/OBJ-20260307-001.json`  
  Packet envelope for the active Gate 0 dispatch.

- `mail/payloads/GATE-0-KICKOFF.md` and `mail/payloads/OBJ-20260307-GATE0-CODEX.md`  
  Supplemental briefing notes. Helpful context, but not the canonical payload contract.

## Rule Of Thumb

Chat is for speed.

QSOP is for truth.
