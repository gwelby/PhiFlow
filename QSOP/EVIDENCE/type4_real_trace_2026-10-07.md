# Type 4 Real SOMA Trace — First Live-Sensor Result

**Date:** 2026-10-07 · **Collector:** Devin (Projects seat)
**Source:** `scripts/collect_soma_trace.py` driving live `soma.py --phiflow --profile harmonic_scan`
on this host (CPU ring fallback; mic absent under WSL).
**Artifact:** `tests/fixtures/soma_live_trace.txt` — REWRITTEN with real data (was May-20 capture).

## Runs

| Run | Samples | obs range | unique obs | L_self (collector) | R_in | R_out | F_model | C_PF | Verdict |
|-----|---------|-----------|------------|--------------------|------|-------|---------|------|---------|
| 30s | 147 | [0, 0.568] | 144 | 0.125 | 0.250 | 0.125 | 0.0156 | 0.0035 | PASS (simple path) |
| 120s | 575 | [0, 0.571] | 561 | **0.058** | 0.176 | 0.058 | 0.0034 | 0.0004 | **FAIL** |

Canonical path (`cargo test --test soma_live_trace_test`, `SelfCorrelation::from_type4_trace` /
`compute_fisher_type4`) on the 147-sample trace: **L_self = 0.0286 ≤ 0.1 → FAIL**
("Real sensor trace does not exhibit self-correlation" — the test's own wording).

## What this means (evidence, not narrative)

1. **Real SOMA telemetry does NOT close the Type-4 self-model loop** under either metric path.
   The synthetic wakeful fixture's L_self = 0.438 overstated what live data delivers by ~7–15×.
2. **More data weakened the result** (0.125 → 0.058): the 30s pass was small-sample noise.
   This is exactly why Codex held the status at synthetic-only.
3. **F_model ≈ 0.003** — the collector's self-model (running mean → binary action) carries
   almost no predictive structure on real sensor data. The model channel is the weak link,
   not the sensor.
4. Prior battery evidence (2026-07-14) passed Phase 3/4 on *synthetic fixtures* and the
   benchmark program's own VM trace — neither is a live sensor-coupled loop.

## Disposition

- C-21/C-23 remain **HOLD — now with a negative real-data point**, not just missing evidence.
- `benchmark_battery` Phase 3 still only exercises synthetic fixtures; the real trace lives in
  `soma_live_trace_test` (which now FAILS on real data — correctly, that's the test working).
- Next honest moves: (a) richer self-model in the capture loop (EMA/structure, not running
  mean), or (b) daemon-coupled trace where `phic` evolves state under live SOMA input
  (the collector's model is Python-side; nothing PhiFlow closes its own loop over sensors yet).
- **Do not tune the model to pass.** If a richer model closes the loop, the evidence report
  must show both results — the FAIL is part of the record.
