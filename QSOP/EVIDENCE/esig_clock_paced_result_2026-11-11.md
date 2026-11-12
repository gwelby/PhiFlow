# ENGINEERED-SIGNAL clock-paced cross-surface probe: Result (2026-11-11)

**Weight: ENGINEERED-SIGNAL.** Not pooled with natural traces or with any
earlier ENGINEERED-SIGNAL run. Rung: producer-tested (one seat, Claude Opus
5.5); re-scored locally from pulled traces, not yet re-scored by a second seat.

- **Prereg:** `QSOP/EVIDENCE/esig_clock_paced_prereg_2026-11-11.md`
- **Seal:** commit `7a9b796` (prereg, probe, `clock_s` sensor and run script, all
  committed before the first run)
- **Artifacts:** `QSOP/EVIDENCE/esig_clock_2026-11-11/{noetguru,8086k}/{ctl,wave}/`
  (`trace.txt`, `prod.log`, `state.json`, `timing.txt`, `score.json`)

## Verdict: PACING NOT THE ONLY ISSUE (as declared)

Gates passed on both surfaces, the surfaces agree to 0.0002, and both L_self
values are above the 0.60 ceiling the prereg set for "pacing artifact confirmed".
Per the declared verdict, the old NoetGuru 0.317 (not the 8086K 0.948) is
now the number that needs explaining.

Sealed prediction for this outcome was 0.08. The point guess (both in
0.20–0.45) was wrong. The named hope ("confirmed") did not land as stated.

## Results

| Surface | Run | Wall (s) | obs range | degenerate | r_in | r_out | L_self | corr(a,o') |
|---|---|---|---|---|---|---|---|---|
| NoetGuru | ctl `--constant 0.5` | 60.250 | 0.500–0.500 | **true** (obs, model, action) | 0 | 0 | 0 | 0 |
| NoetGuru | wave | 60.251 | 0.200–0.900 | false | 0.6811 | 0.9119 | **0.6811** | 0.6460 |
| 8086K | ctl `--constant 0.5` | 60.251 | 0.500–0.500 | **true** (obs, model, action) | 0 | 0 | 0 | 0 |
| 8086K | wave | 60.254 | 0.200–0.900 | false | 0.6809 | 0.9120 | **0.6809** | 0.6459 |

Gates: **G1** timing 60 s ± 3 s: pass (all four within 0.26 s).
**G2** control degenerate: pass on both. The control channel read a live 0.5
every cycle, so it was not a dead channel reading 0.0.
**G3** full-period coverage (obs min ≤ 0.30, max ≥ 0.80): pass on both
(0.200–0.900).

|ΔL_self| between surfaces = **0.0002**. Re-scored on Guru-P1 from the pulled
traces: identical numbers. No processes left on either surface.

## What this shows

1. **Pacing fully explains the cross-surface difference.** Same probe, same
   producer, same 2 s start offset, clock-paced: two machines of very
   different speed give the same L_self to four decimals. The kit rule "pace
   by clock before comparing surfaces" is now measured, not inferred.
2. **The full-period value is 0.68, not ~0.3.** Neither earlier number was the
   "true" full-period value. Both were window artifacts in different
   directions.

## Post-hoc reading (NOT sealed; a hypothesis, not a finding)

The obs channel shows which part of the declared wave each trace sampled
(`0.55 + 0.35·sin(2πt/60)`):

| Trace | obs path | Approx. share of one period sampled | L_self |
|---|---|---|---|
| NoetGuru 2026-11-07, spin-gate | 0.58 → 0.90 → 0.275 | ~0.63 (≈38 s of signal time) | 0.317 |
| 8086K 2026-11-11, spin-gate | 0.61 → 0.90 → 0.85 | ~0.3 | 0.948 |
| Both surfaces 2026-11-11, clock | 0.61 → 0.90 → 0.20 → 0.60 | 1.00 | 0.681 |

L_self for this declared model class (running mean) appears to depend
strongly on how much of the signal's period the window covers. It is high
on a near-monotone stretch, lowest when the window ends partway down the
descent, and in between over a whole period. If that holds, an L_self
magnitude means nothing without its window stated in **signal time**, and
cross-run comparisons need a fixed window/phase, not just a fixed cycle
count.

Also noticed post-hoc: the old NoetGuru run took ~65 s wall but its obs
path covers only ~38 s of signal time. The cause is not established (startup
offset between producer and probe and uneven spin-gate timing are candidates).
The 8086K surface-validation doc called its obs path "monotone". It actually
peaked at 0.90 and ended at 0.85; that doc now carries a correction.

Testing the window-dependence hypothesis would be a new prereg (e.g. sweep
the window length in signal time at a fixed phase, on one surface).

## What this does not change

- The ENGINEERED-SIGNAL lane verdict (CLOSED at engineered weight,
  2026-11-07) stands. Every run here is also non-degenerate and above
  threshold.
- Nothing here bears on natural coupling, C-21/C-23 status, or Type 4
  observer claims.

## Instrumentation notes (disclosed)

- `run_esig_clock.sh` started the producer, slept 2 s, then ran `phic`.
  All four runs began at the same wave phase (first obs 0.6139, or 0.5 in
  the controls).
- 8086K hosted a hackathon job on the RTX 2060 throughout (58% util at
  check). Runs were CPU-only (`CUDA_VISIBLE_DEVICES=""`, `nice -n 19`), one at
  a time. Load stayed at ~1.3/12.
- Remote copies remain in `/tmp/esig_clock_2026-11-11/` (NoetGuru) and
  `~/phiflow_surface/2026-11-11/clock/` (8086K).
