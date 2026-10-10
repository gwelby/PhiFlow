# Type-4 Round 4 Result — Sensorimotor-Coupled Variant (NEGATIVE)

**Date:** 2026-10-10 · **Program:** `examples/type4_soma_motor.phi`
**Preregistration:** `type4_motor_prereg_2026-10-10.md` (committed `a31ca65` BEFORE the run)
**Fixture:** `tests/fixtures/soma_motor_trace.txt` — 600 tuples, live SOMA, ~0.75Hz, 600/600 unique obs

## Canonical result (`from_type4_trace`, threshold unchanged)

| Metric | R3 (expressive) | **R4 (sensorimotor)** |
|--------|-----------------|------------------------|
| R_in   | 0.170           | **0.0959**             |
| R_out  | 0.899           | **1.0000**             |
| L_self | 0.170 ✅        | **0.0959 — OPEN ❌**    |
| F_model| 0.994           | 0.466                  |
| C_PF   | 0.137 (65×null) | 0.0185 (8.8×null)      |

## The environment-coupling diagnostic (preregistered)

`corr(action[t], obs[t+1]) = -0.0052` — **the CPU bursts did not measurably
reach the ring sensor.** The physical link the round was built to test does
not exist at this actuator/sensor surface.

## What R4 actually proved — two negative findings, precisely localized

1. **Actuator-surface mismatch.** A ~65k-iteration single-core interpreter
   burst does not perturb `soma_peak_dbc` — the ring oscillator runs on the
   GPU at 2409Hz; a brief single-core CPU load doesn't move its spectrum
   readout. The actuator exists; the *coupling* doesn't.
2. **Model-channel mismatch.** R_in collapsed 0.170→0.096: `soma_peak_dbc`
   (range 0.09–46.9, spiky) is far more volatile than `soma_presence`
   (0.39–0.60, smoothed). A running mean cannot track it — the model, not
   the action, became the binding joint this round. R_out = 1.0 shows the
   action channel was perfectly expressive and it still failed.

## Status of the construction family (per prereg verdicts)

- Collector proxy (Python): FAIL — L_self 0.029
- Program-carried binary action: FAIL — 0.076 (action starvation)
- Program-carried expressive action: **PASS — 0.170** (observable self-model)
- Program-carried expressive + physical perturbation: **FAIL — 0.096**
  (actuator can't reach the GPU ring; volatile channel defeats running mean)

**Declared scope:** the Type-4 surface at *this* sensor/actuator/model stack
is now mapped: one clean positive (observable self-model), three negatives
each localizing a different joint. Full sensorimotor closure requires either
an actuator that reaches the observable (none available in-language today) or
an obs channel that is both load-sensitive AND trackable — `soma_fan_hz`
(slow thermal response) is the remaining candidate but its lag exceeds the
sample window. Per the prereg: this result is written and tuning stops here
unless a genuinely different actuator surface is designed.
