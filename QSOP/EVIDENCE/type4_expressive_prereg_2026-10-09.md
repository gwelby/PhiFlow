# Type-4 Round 3 Preregistration — Expressive-Action Variant

**Filed:** 2026-10-09, before any round-3 run.
**Ancestor results:** `type4_daemon_trace_2026-10-09.md` — R_out localized as
the bottleneck (0.03–0.08 across three real traces; R_in ~0.2 healthy).

## Hypothesis under test

R_out is low **because the action channel cannot express model state**:
`action = sign(obs − model_mean)` is dominated by obs noise and carries ~1 bit
of model-dependence per cycle. The loop closes only if observable behavior is
a *function of the model* — so round 3 makes the action exactly that:

**`action[t] = model_mean[t]`** — the program acts by publishing its current
prediction (its belief state made observable) BEFORE absorbing obs[t].

## Held constant

- obs channel: `soma_presence` → `soma_peak_dbc*0.05` fallback (unchanged)
- model class: program-carried running mean (unchanged)
- pacing: spin-gate ~100k ≈ 1.16Hz (unchanged), all-fresh samples
- scoring: canonical `from_type4_trace`, threshold L_self > 0.1, 600 samples
- file: `tests/fixtures/soma_daemon_trace.txt` → archived as
  `soma_daemon_trace_r3_<date>.txt` after scoring

## Declared verdicts (before running)

- L_self > 0.1 → loop closes **when behavior expresses the model**. Interpretation
  MUST carry the caveat: this certifies self-consistency of expression
  ("behavior is a faithful function of internal model"), NOT closed-loop
  control of an external process — actions never perturb the sensor stream.
- L_self ≤ 0.1 → even maximal action-expressiveness fails → the metric demands
  something the current obs/action channel structure cannot supply; the
  construction family is then exhaustively mapped at this sensor surface.
- Either way: result recorded verbatim; no in-run tuning.

## Known scope limit (stated up front)

Actions do not feed back into the physical sensor — SOMA is exogenous. So a
PASS demonstrates "program behavior tracks its own evolving model under live
sensation" (Type-4-lite: observable self-model), not full sensorimotor closure.
A full Type-4 claim would need actions that perturb the environment (e.g.,
changing system load the ring oscillator then measures) — noted as the
round-4 candidate if R3 closes.
