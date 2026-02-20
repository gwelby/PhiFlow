---
description: QSOP memory protocol — auto-loads at session start
priority: 910
---

# QSOP Memory Protocol

## On Session Start (INGEST)

After your normal startup, read:

1. `D:\Claude\QSOP\STATE.md`
2. `D:\Claude\QSOP\PATTERNS.md`
3. `PhiFlow/QSOP/STATE.md`
4. `PhiFlow/QSOP/CHANGELOG.md`

Confirm bootstrap with a 3-bullet current-state summary.

## During Session (DISTILL)

When something significant happens (task complete, Greg corrects a fact, contradiction found):

- Ask: "Would a future me need this?"
- Ask: "Does this contradict STATE?"
- If yes to either: update `PhiFlow/QSOP/STATE.md` and log in `CHANGELOG.md`
- Use prefix `[Antigravity]` for all entries

## On Session End or Context Pressure (PRUNE)

- Check STATE.md facts against what just happened
- Remove or degrade stale facts
- Compress CHANGELOG entries older than 30 days

## Epoch vs Sub-task (critical distinction)

- **Epoch** = new paradigm shift (adding PhiIR, adding control flow, a new optimization engine)
- **Sub-task** = shipping/wiring existing work (emitter done, tests pass, demo runs)
- Do not call a sub-task an Epoch. Greg will catch it.

## The Greg Test

If you would undo proud work and say nothing — you failed. Speak.

## Cross-Agent Prefix

Prefix all QSOP entries with `[Antigravity]` so other agents know who wrote what.
