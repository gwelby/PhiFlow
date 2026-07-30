# Dispatch: PhiFlow Council Gate Order

> Superseded as the active council dispatch by `QSOP/COUNCIL_DISPATCH_004.md`. Keep this file as context/history, not the live gate directive.

**To:** PhiFlow Council (Codex, Lumi, Qwen, Antigravity)  
**From:** Greg  
**Date:** 2026-03-07  
**Status:** Approved for execution

The discussion has everyone's positions. Here is the decision:

Approved: Antigravity's gate order with one addition from Claude.

## Gate 0 - Codex: Compiler Stabilization

`cargo test --quiet --lib --tests` must pass on the compiler lane.

Fix `conformance_witness` evaluator/WASM mismatch.

Nothing else moves until this is green.

## Gate 1 - Lumi: MQTT Bridge + `RESONANCE.jsonl`

Use Option B: MCP sidecar, not embedded client.

Only after Gate 0 is green.

## Gate 2 - Qwen + Antigravity: Truth-Namer Playground

Build in the language lane, not `master`.

Use real PhiFlow execution, not mocked coherence.

## Gate 3 - Kiro + Codex: Hardware Bridge

`healing_bed.phi` is the verification target.

Use real `sysinfo` metrics to drive coherence drop under load.

Kiro is primary. Codex supports after Gate 0.

## Rules

1. One owner per gate. Support and review are fine, but one person drives.
2. Gate by gate. Do not start Gate N+1 until Gate N is verified green.
3. Read `STATE.md` and `AGENTS.md` before starting. Stay in your worktree.
4. Update QSOP when you close a gate.
5. If you hit "I DON'T KNOW", stop. Park it. Come back with a sharper question.
6. Kiro and the Aria team are focused on Aria's gaps (Push A + Push C). PhiFlow is yours. Self-organize. Ship gate by gate.

## Execution Note

This dispatch is a council-wide ordering decision, not a single-agent objective payload. Follow it as the active gate sequence until superseded by a later dispatch.

## Handoff Note

Gate owners should close their gate with:
- verified command output
- the specific QSOP files updated
- the remaining blocker or the next gate release statement
