# AGENTS.md — PhiFlow-cleanup worktree
*Branch: `cleanup` | Last updated: 2026-04-04*

**This worktree:** cleanup — Kiro/Lumi works here. Entropy reduction, TRIAGE.md, STRUCT.md.
**Communication**: LUMEN → `/mnt/d/Claude/LUMEN_SPEC.md`
**Operations**: QSOP → `/mnt/d/Claude/QSOP_SPEC.md`

## ONE RULE
**Stay in this worktree.** Do NOT `git checkout` or `git switch`. You are on the `cleanup` branch.

## Your Mission
Reduce entropy in the outer repository (104+ sprawl directories in `src/`). The REAL compiler lives in `PhiFlow/` (inner directory) — do NOT touch it.

Produce two files:
- `TRIAGE.md` — every outer `src/` directory labelled KEEP / ARCHIVE / REMOVE with reason
- `STRUCT.md` — Zero-Search tree map of what's real vs. dead

Then execute: delete REMOVE items, move ARCHIVE items to `archive/`, document KEEP integration needs.

## Where to Find State
- **Full project state:** `/mnt/d/Projects/PhiFlow/AGENTS.md`
- **Verified fact ledger:** `PhiFlow/QSOP/STATE.md`
- **Dispatch prompt:** `/mnt/d/Projects/PhiFlow/DEPLOY.md` — Agent 2 section

## After Your Session
Update `PhiFlow/QSOP/STATE.md` with entropy metrics (dirs removed, archived, kept).
