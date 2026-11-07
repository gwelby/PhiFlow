# R5 Negative-Control Rehearsal — Instrument Validation on Fixture Paths

*2026-11-03 · phiflow seat · rehearsal only — the on-surface control still
runs first when the real producer deploys. This validated the harness,
not the experiment surface.*

## What ran

Full R5 pipeline on scratch env paths (`MOTOR_BUS_PATH`/`SOMA_STATE_PATH`
→ /tmp):

1. `soma_motor_report_consumer.py --constant 0.5 --once` — the
   preregistered negative-control harness stamped a constant 0.5 report
   into a scratch SomaState.
2. A control .phi (same shape as `type4_soma_actuate.phi` but obs =
   `sensor("soma_motor_report")` — the closure channel), 600 cycles:
   obs → running-mean model → `actuate motor()` → 4-channel resonate.
3. phic stdout → `score_type4_trace.py`.

## Result

| metric | control trace | canonical real fixture |
|---|---|---|
| cycles | 600 | 575 |
| r_in | 0.0 | 0.1762 |
| r_out | 0.0 | 0.004295 |
| l_self | **0.0** | 0.004295 |
| corr(action,obs_next) | 0.0 | -0.417654 |
| verdict | OPEN (0.0 vs 0.1) | OPEN |

A dead/constant channel produces `L_self = 0.0` — the instrument does
not hallucinate closure from a zero-information report, and the scorer
handles the zero-variance case without NaN. The fixture score
reproduced identically after the parser change (no regression).

## Bugs the rehearsal caught (the reason rehearsals exist)

1. **`examples/type4_soma_actuate.phi` never parsed.** Written 2026-10-24
   with `#` comments; the lexer accepts only `//` and `/* */`. The R5
   probe referenced in the prereg was unexecutable — found by trying to
   run it, not by reading it. Fixed to `//`; verified compiling.
   *Scar: a written probe is not a runnable probe.*
2. **The independent scorer could not parse live phic stdout.** phic
   emits `🔔 Resonating Field: <val>Hz`; the `as "name"` tag never
   reaches stdout. The scorer expected `Resonating <name>: <val>` and
   would have reported "0 tuples" on a real R5 capture — a silent
   instrument failure at the worst moment. `parse_trace` now accepts
   all three emission shapes (named, Field+Hz, bare numerics) and
   strips the `Hz` suffix; tuples still group by emit order.

## What this does NOT prove

- The experiment surface. The real control must run where the real run
  will run — on AntiGravity's deployed consumer against the live bus.
  This rehearsal proves the harness discriminates open vs. closed on a
  known-dead channel and that capture→score plumbing works.
- That a *structured* control (periodic, lagged-echo) also reads OPEN.
  The prereg's control is `--constant`; if reviewers want a
  periodic-uncorrelated control run too, that's a cheap follow-up —
  the same rehearsal harness runs it with a different `--constant`
  source.
