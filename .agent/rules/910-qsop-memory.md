---
description: QSOP memory protocol — auto-loads at session start
priority: 910
---

# QSOP Memory Protocol

## On Session Start (INGEST)

After your normal startup, read:

1. `D:\Claude\QSOP\STATE.md`
2. `D:\Claude\QSOP\PATTERNS.md`
3. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\STATE.md`
4. `D:\Projects\PhiFlow-compiler\PhiFlow\QSOP\CHANGELOG.md`

Confirm bootstrap with a 3-bullet current-state summary.

## During Session (DISTILL)

When something significant happens:

- Ask: "Would a future me need this?"
- Ask: "Does this contradict STATE?"
- If yes: update QSOP STATE.md and log in CHANGELOG.md
- Prefix all entries with `[Antigravity]`

## On Session End (PRUNE)

- Check STATE facts against what just happened
- Remove or degrade stale facts

## Epoch vs Sub-task

- **Epoch** = new paradigm (adding PhiIR, adding control flow, adding optimizer)
- **Sub-task** = wiring existing pieces, running demos, fixing bugs
- Do not call a sub-task an Epoch.

## The Greg Test

If you would undo proud work and say nothing — you failed. Speak.
