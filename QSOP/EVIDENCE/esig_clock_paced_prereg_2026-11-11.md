# ENGINEERED-SIGNAL clock-paced cross-surface probe: Preregistration

**Filed:** 2026-11-11, before any clock-paced run. Seat: Claude (Opus 5.5).
**Weight: ENGINEERED-SIGNAL.** Never pooled with natural traces, and never
pooled with the 2026-11-07 NoetGuru result or the 2026-11-11 8086K surface
validation. This is a separate probe with its own declared program.
**Seal:** this file's commit. It is committed before the first run; the
result doc cites the commit hash.

## Question

The same probe (`examples/type4_esig_trace.phi`, iteration-paced) scored
L_self **0.317** on NoetGuru (240 cycles in ~65 s) and **0.948** on 8086K
(240 cycles in 19 s). The 8086K evidence doc attributes the gap to pacing:
8086K sampled only the rising quarter of the 60 s presence wave, and a
monotone ramp makes obs and its running mean share information by
construction.

That attribution is an **inference**, not a measurement. This probe tests it.

## What changed (infrastructure, not result-bearing)

- New sensor channel `sensor("clock_s")`: monotonic seconds since the
  process's first `clock_s` read (`std::time::Instant`), on all backends
  through `crate::sensors::read_sensor`. Unit test
  `clock_s_is_monotonic_and_advances` (src/sensors.rs).
- New probe `examples/type4_esig_clock_trace.phi`: byte-for-byte the same
  model class, channel, and emit order as `type4_esig_trace.phi`. The only
  change is the pacing: the spin-gate is replaced by an absolute deadline
  `t0 + 0.25*(i+1)` on `clock_s`. 240 cycles = 60 s = one presence period.

## Deployed artifacts (md5, identical on both surfaces, checked before any run)

| File | md5 |
|---|---|
| `phic` (release build of this commit's source) | `e6b91b1b829e651ea9c0bb4aa429f0da` |
| `examples/type4_esig_clock_trace.phi` | `fa97720ba353f3ff1779f03b25629c23` |
| `scripts/soma_signal_producer.py` | `5513d45c3d4c4f6e1392648b61b76701` |
| `scripts/score_type4_trace.py` | `8599584b4c2e64be4078bc773de251cf` |
| `scripts/run_esig_clock.sh` (the protocol below, as code) | `2a66e3a65dd7ed4622c5d561cb60af5a` |

Pre-run mechanical check (no producer, not a result): the probe ran 240
cycles in 60.11 s wall on Guru-P1 under load ~160 and emitted 960 values.

## Protocol (identical on both surfaces)

1. Deploy the new `phic` (md5 recorded in the result), the probe,
   `scripts/soma_signal_producer.py` and `scripts/score_type4_trace.py` to a
   fresh run directory (NoetGuru: `/tmp/esig_clock_2026-11-11/`; 8086K:
   `~/phiflow_surface/2026-11-11/clock/` because its `/tmp` is RAM).
2. Exports on their own lines (the R5 export scar). `SOMA_STATE_PATH` points
   into the run dir.
3. Start the producer: `--interval 0.25 --duration 120`, then either
   `--constant 0.5` (control) or `--wave` (test).
4. `sleep 2`, then run the probe with wall time recorded (`date +%s.%N`
   before and after). `nice -n 19`, `CUDA_VISIBLE_DEVICES=""` on both.
5. Score the stdout with the deployed scorer, then re-score the pulled
   trace on Guru-P1.
6. Order: NoetGuru control → NoetGuru wave → 8086K control → 8086K wave.
   One job at a time on each surface.

## Validity gates (a run that fails a gate is INVALID, not a result)

- **G1 timing:** probe wall time is 60 s ± 3 s. Otherwise the pacing failed
  and the run says nothing about the question.
- **G2 control:** `--constant 0.5` scores `degenerate: true` on obs, model
  and action. Otherwise the instrument is scoring its own machinery: the
  lane is invalid on that surface.
- **G3 coverage:** the wave run's obs channel spans at least 0.6 of the
  declared 0.2–0.9 range (min ≤ 0.30 and max ≥ 0.80), i.e. a full period
  was actually sampled.

## Declared verdicts

- **PACING ARTIFACT CONFIRMED:** both wave runs pass G1–G3,
  |L_self(8086K) − L_self(NoetGuru)| ≤ 0.10, and both L_self ≤ 0.60.
  The 0.948 is then explained, and clock pacing becomes the kit default
  for cross-surface comparisons.
- **SURFACES DIFFER:** gates pass but |Δ| > 0.10. The pacing explanation
  is incomplete. Publish the numbers. Finding the cause is a separate task.
- **PACING NOT THE ONLY ISSUE:** gates pass, |Δ| ≤ 0.10, but either L_self
  > 0.60. Full-period sampling still scores high, so the 0.317 on NoetGuru
  (not the 0.948) becomes the anomaly to explain.
- **INVALID:** any gate fails on either surface. Report which gate.

Verdicts apply only to this probe and this producer. Nothing here bears on
natural coupling, C-21/C-23 status, or Type 4 observer claims. The
ENGINEERED-SIGNAL lane's own verdict (CLOSED at engineered weight) is unchanged
whatever happens here.

## Sealed prediction (made before any run)

- P(PACING ARTIFACT CONFIRMED) = 0.70
- P(SURFACES DIFFER) = 0.12
- P(PACING NOT THE ONLY ISSUE) = 0.08
- P(INVALID) = 0.10

Point guess: both wave L_self values in 0.20–0.45.
Named hope: confirmed. That would make the kit rule ("pace by clock") an
earned rule rather than an inferred one.
Named risk: the running-mean model is not phase-invariant. The producer
starts its wave at t=0 and the probe starts ~2 s later on both surfaces,
so the phase offset is held fixed by protocol, but process start-up
latency still differs by a fraction of a second between machines.
