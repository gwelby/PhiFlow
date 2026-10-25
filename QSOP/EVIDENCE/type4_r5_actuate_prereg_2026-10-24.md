# Type-4 Round 5 Preregistration — Engineered Closure via `actuate motor()`

**Filed:** 2026-10-24, before any round-5 run.
**Ancestor results:**
- R1 collector FAIL (0.029) · R2 binary-action FAIL (0.076)
- R3 expressive-action PASS (L_self=0.170, R_out=0.899) — scope limit: action
  never perturbed the sensor
- R4 motor-burst FALSIFIED — `model → action → environment → obs` showed no
  natural coupling at this surface: the CPU-work burst did not reach
  `soma_peak_dbc` (see `type4_motor_trace_2026-10-10.md` and the sustained
  all-core probe: coupling absent at maximum tested load).

## What R5 changes

R4 asked whether action *naturally* perturbs the observable and the answer was
no. R5 does not retry the natural channel. It closes the loop through the
**explicit engineered limb** added in `da61589`: `actuate motor("channel",
expr)` writes a bounded tuple to `MOTOR.jsonl`; a consumer process reads the
bus and writes `soma_motor_report` into the state SOMA exports; the next
observation cycle reads it back through `sensor()`.

The loop:
```
model → actuate motor("soma_motor", a) → MOTOR.jsonl
      → consumer (soma_motor_report producer) → SOMA state
      → sensor("soma_motor_report") → model
```

Every edge is explicit and inspectable. This is **engineered closure**, and
the claim language must stay honest about it: a positive R5 result
demonstrates that the stack *can* carry a self-model loop end-to-end through a
declared bus — it does not resurrect the falsified natural-coupling claim.
`CLAIMS.md` wording for any R5 writeup must say "engineered closure" and cite
R4's falsification in the same breath.

## Held constant

- model class: program-carried running mean (unchanged from R2–R4)
- action trace value: `model_mean` at decision time
- scoring: canonical `from_type4_trace`, L_self > 0.1, 600 samples
- pacing: same spin-gate cadence as R4 where the program allows
- program: `examples/type4_soma_actuate.phi` (already exists)

## Channel declarations (required)

- **efferent:** `actuate motor("soma_motor", model_mean)` → `MOTOR.jsonl`
  (bounded, `motor_bus.rs` schema unchanged)
- **consumer:** an external `soma_motor_report` producer — currently
  AntiGravity-side, **not yet running**. The prereg declares its contract
  before it exists:
  - reads `MOTOR.jsonl` tail
  - writes `soma_motor_report: f64` into the SOMA-exported state with a
    timestamp no older than one sample window
  - writes nothing when the bus is silent (a dead bus must read as absent,
    not as a held value — same fail-closed rule as ClaimsDrift)
- **afferent:** `sensor("soma_motor_report")` as the obs channel. If the
  report field is absent, `sensor()` returns None → the run scores as
  "closure channel absent", a declared outcome, not a bug.

## New diagnostics (declared)

1. `corr(action[t], obs[t+1])` — direct signal check on the engineered edge,
   same as R4's diagnostic but now over the bus path.
2. **Bus latency histogram** — `actuate` timestamp vs report timestamp; the
   loop can only be self-modeling if obs[t+1] carries the action written at
   t, not a stale replay.
3. **Negative control** — one run with the consumer writing a *constant*
   report. If L_self still exceeds 0.1, the metric is scoring the program's
   own periodicity, not the loop; that result invalidates the surface, not
   the language.

## Amendment A (2026-10-24, same day, before any run) — honesty protocol

Added in response to a self-question recorded in `/mnt/d/Devin/MYWISH.md`
(phiflow seat ME-Time): *"do I want Type-4 to be yes, or want the truth
regardless?"* — which is unanswerable by declaration. These amendments make
the run an instrument for that question too, by fixing the decision points
where the two wants diverge **before** they can be rationalized.

1. **Sealed prediction.** The experimenters's honest probability split and
   declared hope are hash-sealed with `Tools/blind.py`
   (`logs/blind/phiflow-r5-verdict.ndjson`, seat `projects`). Post-hoc
   position drift shows as HASH MISMATCH. Prediction: P(closure yes)=0.35,
   P(instrument artifact)=0.20, P(L_self≤0.1 live bus)=0.35,
   P(never runs)=0.10. Declared hope: yes — named, not hidden.
2. **Control runs FIRST.** The constant-report negative control executes
   before the real run. If it crosses 0.1 the surface is invalid and the
   real run does not happen — the report publishes the invalidation.
   (Wanting-yes runs the experiment first and relitigates the control
   after; this ordering removes the temptation.)
3. **Independent scorer.** `L_self`/`corr(action,obs_next)` on the R5 trace
   gets recomputed by a second seat (Kiro or Codex) from the raw trace
   fixture — the primary number must not exist only in the
   experimenter's hands.
4. **Disclosure obligations.** Published regardless of outcome: control
   result, raw `L_self`/`R_in`/`R_out`, bus-latency histogram, and any
   aborted or crashed runs. Selective disclosure is where wanting-yes
   leaks; the list is fixed here so nothing is "decided later".

## Declared verdicts (before running)

- **L_self > 0.1 AND corr(action,obs_next) meaningfully > 0 AND control run
  fails to cross** → engineered-closure self-model confirmed at this stack.
  Claim tier: EMPIRICAL, engineered-closure only.
- **L_self > 0.1 but control run also crosses** → instrument artifact; the
  scorer sees program periodicity. Downgrade to UNDERDETERMINED, do not
  publish a positive.
- **L_self ≤ 0.1 with a live bus** → the loop is not self-modeling even when
  explicitly closed; report and stop tuning at this surface.
- **consumer never ships** → R5 unrun; this document records the blocked
  protocol, not a result.

## What is NOT claimed

R5 is not evidence that natural sensor-to-motor coupling exists anywhere in
this stack — R4 falsified that. R5 tests only whether the declared bus can
carry the loop when every edge is engineered.

Run (once the consumer is live):
`SOMA_STATE_PATH=... MOTOR_BUS_PATH=... phic --measure --max-steps 15000000000
examples/type4_soma_actuate.phi`
