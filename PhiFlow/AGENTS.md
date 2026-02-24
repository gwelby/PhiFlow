# PhiFlow Inner Agent Guide

Scope: `D:\Projects\PhiFlow\PhiFlow` and all children.

## Startup Order

Read in this order before edits:
1. `QSOP/STATE.md`
2. `QSOP/PATTERNS.md`
3. `QSOP/CHANGELOG.md`
4. `CLAUDE.md`
5. `LANGUAGE.md`

## Operating Contract

- Stay inside this worktree. Do not touch sibling worktrees.
- Keep changes small and testable.
- Run the smallest relevant verification command before finishing.
- If you find architecture drift, write it to `QSOP/STATE.md`.
- If you find a repeat bug pattern, write it to `QSOP/PATTERNS.md`.
- Every completed objective gets a dated `[Codex]` entry in `QSOP/CHANGELOG.md`.

## Team Handoff Format

When handing off to another agent, include:
- Objective ID
- Branch/worktree
- Files changed
- Verification command + result
- Risks or unknowns
