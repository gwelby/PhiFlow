# Type-4 Daemon-Coupled Trace — Preregistration

**Filed:** 2026-10-09, before any daemon-coupled run.
**Predecessor evidence:** `type4_real_trace_2026-10-07.md` — host-carried Python
running-mean on 575 live SOMA samples gave canonical L_self = 0.0286 ≤ 0.1 →
**FAIL** (synthetic wakeful fixture had predicted 0.438 — fixture is not evidence).

## What this experiment changes (and only this)

The claim C-21 requires *the program* to close the self-model loop. The failed
run placed the model in a Python collector — host-carried state, program absent
from the loop. `examples/type4_soma_daemon.phi` moves the model into the
program's own mutable agent state, persists across daemon yields/restarts via
`--state-path`, reads obs from live SOMA inside the language runtime
(`witness sensor(...)`), and emits the canonical STEP|OBS|MODEL|ACTION tuple
through `resonate`.

**Held constant:** obs channel contract (soma_presence, peak_dbc fallback),
running-mean model class, binary action rule, canonical `from_type4_trace`
scoring, threshold L_self > 0.1, ≥600 cycles (> the 575 baseline).

**Not controlled:** model class is still running-mean — deliberately, for clean
A/B against the collector FAIL. If the loop still opens, the result isolates
*where the model lives* as not-the-weak-link and the question moves to model
class (or obs surface richness).

## Declared verdicts (before running)

- `from_type4_trace` L_self > 0.1 → C-21 earns its first real POSITIVE data
  point (moves HOLD → candidate CONFIRMED pending replication)
- L_self ≤ 0.1 → second independent real-trace negative; rules out
  model-location as the cause; next variable is model class or sensor surface
- Flat obs (dead presence AND dead fallback) → sensor-surface finding, not a
  Type-4 verdict; retest on a host with live mic/ring variance

## Rules for the run

- No threshold or model tuning inside the run — both outcomes land in STATE.md
- Trace saved to `tests/fixtures/soma_daemon_trace_<date>.txt`
- Synthetic fixtures stay labeled synthetic; this run supersedes nothing until
  its result is recorded
