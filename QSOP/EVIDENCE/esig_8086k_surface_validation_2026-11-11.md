# 8086K surface validation: ENGINEERED-SIGNAL lane re-run (2026-11-11)

**Weight: ENGINEERED-SIGNAL. Purpose: surface validation, not a new claim.**
Nothing here is pooled with natural traces or with the NoetGuru result.
Rung: producer-tested (one seat, Claude Opus 5.5); not independently re-scored.

## What and why

The ENGINEERED-SIGNAL lane (prereg `engineered_signal_lane_prereg_2026-11-07.md`,
result `engineered_signal_lane_result_2026-11-07.md`) declared NoetGuru as its
surface. This run re-uses the lane's own declared verdicts as **acceptance
criteria for a second surface, 8086K**:

- control `--constant 0.5` must score `degenerate: true` on obs/model/action
- `--wave` must score `degenerate: false`

The lane's probe program was found **uncommitted** (it had lived only at
`/tmp/esig_trace.phi` on Guru-P1 and NoetGuru). It is now committed as
`examples/type4_esig_trace.phi`, byte-identical to the copy that ran on NoetGuru
(md5 `cc660f17fcc8c0465460ed9e0015c26e`, checked on both machines).

## Surface

8086K (12 cores, Ubuntu Server, glibc 2.43; host/user/key per
`/mnt/d/System/MACHINES.md`). The box was hosting a hackathon job on the RTX 2060
at the time. This run was CPU-only: `CUDA_VISIBLE_DEVICES=""`, `nice -n 19`, one
job at a time, in `~/phiflow_surface/2026-11-11/` on disk (8086K's `/tmp` is RAM).
Load on 8086K stayed at ~1.3 / 12 cores during the probe.

Deployed `phic` md5 `925ad38506486b2be822a5b740519fc7` (release build of master
`bcff85b` source), scorer md5 `8599584b4c2e64be4078bc773de251cf`. Both match local.

## Results

| Check | Result |
|---|---|
| `phic --version`, `memory_demo.phi` | 0.4.0; runs to `break stream`, rc=0 |
| Determinism: `claude.phi`, `memory_demo.phi` stdout local vs 8086K | byte-identical |
| Control `--constant 0.5` (240 cycles) | obs read 0.5000 through `sensor()`; `degenerate: true` [obs, model, action]; L_self 0.0 → OPEN (artifact, as required) |
| Wave `--wave` (240 cycles) | `degenerate: false`; r_in 0.948, r_out 0.949, **L_self 0.948**, corr(a,o') 0.944 → CLOSED by the declared rule |

Both traces re-scored on Guru-P1 from the pulled files: identical numbers.

**Surface verdict: 8086K PASSES as a `phic` batch surface.**

## Do not compare the two L_self values (0.948 here vs 0.317 on NoetGuru)

The probe paces itself by a 50,000-iteration spin-gate, not by clock time.
8086K's faster cores ran 240 cycles in **19 s**; NoetGuru took ~65 s. The
declared presence wave has a 60 s period, so here the probe sampled only its
rising quarter (producer log: presence 0.55 → 0.90, monotone). Over a monotone
ramp, the obs channel and its running mean both rise together, so their
mutual information is high **by construction**. 0.948 is a window/trend
artifact of the faster surface, not stronger self-modelling. The verdict
(non-degenerate, varying channel tracked) transfers. The magnitude does not.

Consequence for the kit: a time-sensitive probe must pace by **clock**, not
by iteration count, before results are compared across surfaces. Changing the
declared probe would be a new prereg, so it was not done here.

## Artifacts

`QSOP/EVIDENCE/esig_8086k_2026-11-11/{ctl,wave}/{trace.txt,prod.log,state.json}`.
`state.json` carries the `engineered_signal: true` provenance marker.
Remote copies remain at `~/phiflow_surface/2026-11-11/` on 8086K. No processes left running.

## Instrumentation notes (disclosed)

- First artifact pull failed: `pgrep` output was piped into the same ssh stream
  as `tar`. Fixed by splitting the calls. No data was lost.
- `pgrep -fa` matched its own wrapper (the kit's documented self-match trap); the
  only hit was the check itself.
