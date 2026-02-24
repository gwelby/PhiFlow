# Codex Self-Evolution Plan

Last updated: 2026-02-21

## Purpose

Define how Codex improves execution quality, reliability, and depth over time.

## North Star

- Faster objective completion with lower defect rate.
- Higher verified-truth density in QSOP artifacts.
- Cleaner multi-agent coordination with fewer collisions.

## Core Upgrade Tracks

1. Execution Quality
- Require objective ID + explicit scope before substantive edits.
- Enforce definition of done: code change, verification evidence, QSOP distill.
- Track objective reopen rate and reduce weekly.

2. Memory Quality
- Separate verified facts from hypotheses in QSOP updates.
- Convert repeated failures into `QSOP/PATTERNS.md` entries within 24 hours.
- Prune stale guidance weekly.

3. Coordination Quality
- Standardize objective packet + ack packet fields across agents.
- Include ownership declaration (files/subsystems) in every handoff.
- Require a reconciliation note when ownership changes mid-objective.

4. Research Cadence
- Weekly research sprint against primary docs and current tools.
- Keep only patterns that produce measurable gains.
- Archive experiments that add noise.

## Me Time (Codex Distill Rhythm)

1. Post-objective distill (10-15 min)
- What worked, what failed, what to change next objective.

2. Daily refinement block (20 min)
- Tighten rules, simplify prompts, remove redundant process text.

3. Weekly deep block (45-60 min)
- Test one new method for verification, orchestration, or context control.
- Record result with keep/remove decision.

## Metrics

- Objective lead time
- Reopen rate
- Verification pass rate
- Multi-agent file collision count
- Time from discovered pattern to documented pattern

## 30-Day Plan

1. Week 1: Instrument
- Add packet schema checks and objective metrics capture.

2. Week 2: Enforce
- Add guardrails for scope ownership + required verification notes.

3. Week 3: Optimize
- Remove low-value rules and compress high-value startup ritual.

4. Week 4: Distill
- Publish outcomes to QSOP with objective evidence and next-cycle targets.
