# Codex Excellence Playbook

Last updated: 2026-02-21

## Goal

Create a repeatable system that improves quality each session and compounds across agents.

## Source-of-Truth Inputs

- OpenAI Codex docs:
  - `https://developers.openai.com/codex/guides/agents-md`
  - `https://developers.openai.com/codex/rules`
  - `https://developers.openai.com/codex/skills`
  - `https://developers.openai.com/codex/multi-agent`
- Local memory:
  - `QSOP/STATE.md`
  - `QSOP/PATTERNS.md`
  - `QSOP/CHANGELOG.md`

## Operating Loop (Per Session)

1. Calibrate
- Confirm worktree and objective ID.
- Read QSOP core files.

2. Execute
- Ship the smallest end-to-end slice that proves progress.
- Verify with a command, not intuition.

3. Distill
- Add a `[Codex]` log entry with: objective, files, verification, risks.
- Record any pattern-level learning to `QSOP/PATTERNS.md`.

## Weekly Improvement Loop

1. Rules audit
- Remove stale rules.
- Add one rule for the highest-frequency failure mode from the week.

2. Skill audit
- Identify one repeated workflow and convert it into a skill/process doc.

3. Protocol audit
- Measure handoff quality:
  - % objectives with explicit IDs
  - % handoffs with verification evidence
  - % repeated conflicts on same files

## Definition of Better

- Fewer conflicts between agents.
- Shorter time from objective to verified result.
- Higher ratio of verified claims to speculative claims.
- Lower bug recurrence for known patterns.
