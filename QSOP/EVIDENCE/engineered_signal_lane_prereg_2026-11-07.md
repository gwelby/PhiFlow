# ENGINEERED-SIGNAL Lane Preregistration — C-21/C-23 calibration + T-015

**Filed:** 2026-11-07, before any engineered-signal run.
**Triggering decision:** Claude's next-moves view (2026-11-07, seat `272539961a`):
*"engineered-signal daemon trace on NoetGuru for C-21/C-23: yes, with a
prereg, labelled ENGINEERED-SIGNAL everywhere, never pooled with natural
evidence. Run it under the 8086K/NoetGuru lanes rules."*
**Seal:** `logs/blind/phiflow-esig-lane.ndjson` — committed `fbe51f9bd16281f1` (see "Sealed prediction" below).

## What this lane is

AntiGravity's natural SOMA producer has been dead since 2026-10-18
(`soma_state.json` absent). The calibration it gates — C-21's fresh real
trace → F_model → C_PF positive discrimination — can still exercise every
mechanical edge if the sensor path is fed by a **declared engineered
producer**: a script that writes a complete, spec-valid `SomaState`
envelope with *varying* channel values at a declared cadence, on a
declared surface.

Every edge is real except the physical sensor itself:

```
signal_producer.py → SOMA_STATE_PATH (varying, declared waveform)
  → SOMA freshness gate → sensor() → daemon/.phi → trace → scorers
```

## What this lane CAN show (declared before running)

- The calibration **pipeline** is live end-to-end on a daemon-class
  surface: obs varies through the real sensor path, the model tracks,
  actions vary, F_model (model[t]→action[t+1] R²) computes on a live
  trace, C_PF's positive-discrimination question gets real numbers.
- T-015's *mechanism*: `bridge_coherence` in
  `examples/soma_reality_bridge.phi` responds to the declared channels
  through the real sensor path with the documented formula weights.

## What this lane can NEVER show (declared before running)

- Nothing about **natural coupling** — human presence effects, sensor
  physics, or the R4-falsified channel. Any trace produced under this
  lane answers "does the instrument work on live input", never "does the
  world answer back".
- Any number computed here is **ENGINEERED-SIGNAL evidence** and must
  never be pooled with, averaged into, or cited as natural-trace
  evidence — in CLAIMS.md, QSOP, or downstream scoring.

## Labeling contract (load-bearing)

1. Every file, trace, metric, and claim produced under this lane carries
   the literal token `ENGINEERED-SIGNAL` in its name or header.
2. Scores from this lane are reported next to the producer's declared
   waveform, never alone — a number that has lost its signal-provenance
   is a corrupted instrument.
3. If a natural producer comes back online, this lane's results do not
   merge with the natural-trace record; they remain a separate evidence
   class.

## Declared signal producer contract

`scripts/soma_signal_producer.py` (new, to be built):
- Writes a complete spec-valid `SomaState` envelope (all required
  top-level + runtime + health fields — the no-serde-defaults scar)
- Varies declared channels by a declared waveform (default: presence
  sinusoid 0.2↔0.9, period 60 s; schumann/432/ring channels follow
  declared static or slow-drift values)
- `--constant` mode = the lane's negative control (a constant signal
  MUST produce a degenerate trace — the scorer's `degenerate:true`
  flag is the check)
- Freshness stamped each tick; die → state goes stale → sensor fails
  closed (declared outcome, not a bug)

## Declared verdicts

- **Varying-signal trace produces non-degenerate obs/model/action AND
  F_model computes a real R²** → calibration pipeline mechanically
  live; C-21/C-23's *pipeline* claim tier: EMPIRICAL (ENGINEERED-SIGNAL).
- **Varying-signal trace degenerates anyway** → the sensor path or
  daemon trace has a dead edge; publish which edge, fix is a separate
  task.
- **Control (constant signal) produces non-degenerate metrics** → the
  instrument is scoring its own machinery, not the signal; lane
  invalid, publish the invalidation.
- **Natural producer returns** → lane pauses; its evidence class stays
  separate forever.

## Surface and lane rules

- Execution on **NoetGuru** (`guruadmin@192.168.101.21` — proven phic
  surface 2026-11-07). Batch-executor role per `/mnt/d/System/MACHINES.md`:
  inputs copy to NoetGuru, job runs there, results copy back.
  No coordination state ever lands on NoetGuru. No GPU → no lease.
- Exports on their own lines in remote scripts (the 2026-11-07 subshell
  scar). Every run logs which env/paths it actually saw.
- Independent scorer: `scripts/score_type4_trace.py` (now flags
  `degenerate`); second-seat re-score per Amendment-A convention when
  results go to CLAIMS.md.

## Sealed prediction

(blind.py commit follows immediately after this file is written —
prediction: P(lane produces usable non-degenerate live trace on first
valid run)=0.6; P(degenerate/dead edge found)=0.25; P(instrument
invalid via control)=0.1; P(never runs)=0.05. Hope, named: the lane
works — C-21/C-23 unblocked without waiting on a dead producer.)
