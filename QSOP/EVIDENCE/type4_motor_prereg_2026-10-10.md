# Type-4 Round 4 Preregistration — Sensorimotor-Coupled Variant

**Filed:** 2026-10-10, before any round-4 run.
**Ancestor results:** R1 collector FAIL (0.029) · R2 binary-action FAIL (0.076)
· R3 expressive-action PASS (L_self=0.170, R_out=0.899 — observable self-model).
R3's declared scope limit: action never perturbed the sensor. Round 4 removes
that limit.

## The one change that matters

**Actions now touch the environment SOMA measures.** Each emitted cycle, the
program performs a real CPU work burst with size ∝ its model state
(`while i < action*5000` — ~65k interpreter iterations, a sub-second
single-core load on the host), then
SOMA's ring oscillator — which runs on this host — can physically register the
perturbation in the next sample window. For the first time:
`model → action → environment → obs → model` is a physical loop, not a log.

## Channel change (declared, required)

obs := **`soma_peak_dbc`** (ring-spectrum peak amplitude — the load-sensitive
exported channel) instead of soma_presence. Presence is not physically
addressable by CPU work; peak_dbc is. Fallback: none — if peak_dbc is flat,
that is the "coupling channel absent" finding.

## Held constant

- model class: program-carried running mean (unchanged)
- action trace value: `model_mean` at decision time (same as R3)
- pacing: ~100k-spin gate + burst work (~0.6–1Hz effective)
- scoring: canonical `from_type4_trace`, L_self > 0.1, 600 samples
- fixture: `tests/fixtures/soma_motor_trace.txt` (new file — R2/R3 preserved)

## New diagnostic (declared)

`corr(action[t], obs[t+1])` computed post-hoc — the direct test of whether the
action→environment link carries signal. Reported alongside L_self either way.

## Declared verdicts (before running)

- **L_self > 0.1 AND corr(action,obs_next) meaningfully > 0** → sensorimotor-
  coupled self-model — strongest Type-4 reading available on this stack.
- **L_self > 0.1 but corr ≈ 0** → loop closed on expressiveness alone; the
  coupling channel is dead at this surface (the burst doesn't reach the
  observable) — a different, equally valuable finding.
- **L_self ≤ 0.1** → the construction family is mapped at this surface;
  write it and stop tuning.

Run: `SOMA_STATE_PATH=... phic --measure --max-steps 15000000000
examples/type4_soma_motor.phi`
