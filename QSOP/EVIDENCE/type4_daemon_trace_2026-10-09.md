# Type-4 Daemon-Coupled Trace — Round 2 Evidence

**Date:** 2026-10-09 · **Runs by:** Devin ∇λΣ∞ · **Predecessor:** `type4_real_trace_2026-10-07.md` (collector FAIL: L_self 0.029)
**Prereg:** `type4_daemon_prereg_2026-10-09.md` (incl. Amendment 1 — pacing changes declared before results known)

## What changed vs the collector run

`examples/type4_soma_daemon.phi` puts the self-model **inside the program**:
mutable agent state carries the running mean, `witness sensor("soma_presence")`
reads live SOMA each loop, the binary branch is the program's own action.
SOMA was already live on this host (GPU ring, 5Hz fusion, session since Oct 6).

## Canonical results (all scored by `from_type4_trace` — the FAIL test's path)

| Run | Samples | Eff. rate | Unique obs | R_in | R_out | L_self | Verdict |
|-----|---------|-----------|-----------|------|-------|--------|---------|
| Collector (host proxy, Oct 7) | 575 | ~4.8/s | 561 | 0.250 | 0.029 | **0.0286** | OPEN |
| Daemon v2 — 57/s oversample | 600 | 57/s | 42 | 0.236 | 0.064 | **0.064** | OPEN (artifact control) |
| Daemon v3 — ceiling-stopped | 434 | 1.4/s | 434/434 | 0.209 | 0.055 | **0.055** | OPEN |
| **Daemon v4 — full protocol** | **600** | **1.16/s** | **600/600** | **0.186** | **0.076** | **0.076** | **OPEN — third real-trace negative** |

The v4 run is the complete protocol: 600 clean samples over 515s, every read a
fresh fusion write (600/600 unique), actions split 180/420 — the program's
branch genuinely moved. `tests/fixtures/soma_daemon_trace.txt`.

**Interpretation:** moving the model into the program + clean sampling roughly
doubled R_out (0.029→0.076) — the coupling direction is right but
insufficient. R_in softened slightly (0.25→0.19). The bottleneck is now
unambiguous: **the action class** — a binary `obs < mean` branch cannot express
enough model-dependence. Any next run changes the action/model class as a NEW
preregistered experiment; tuning within this design is forbidden.

## What the data says

- **R_out is the bottleneck, consistently.** Model tracks obs history (R_in
  ~0.21–0.25 in both live runs) but the model→action link carries almost no
  predictive signal (R_out 0.03–0.06). The binary `obs < model_mean` action
  barely depends on model state. **Model location (host vs program) is NOT
  the weak link — the action/model class is.**
- The canonical scorer resisted the sub-fusion sampling artifact (L_self
  stayed 0.064 on a 42-unique trace) — good property, worth noting.
- **Metric-path discrepancy flagged:** `--measure` reported l_self=0.71 on the
  oversampled trace vs canonical 0.064. `--measure` is telemetry; the
  `from_type4_trace` test path is the authority. Difference worth a proper
  look — two self-correlation implementations disagree by 11×.

## Defects found along the way (recorded, not fixed)

1. **Hypervisor wedge:** `phic --daemon` stalls ~2 council cycles in — burns
   ~1 core, zero I/O, stream frozen (not ledger contention; solo council
   wedges identically). Daemon mode is not currently a usable runner.
2. **Evaluator 1e9 step ceiling:** default `max_steps` kills long spins;
   `--max-steps 0` lifts it (documented flag behavior).
3. **RESONANCE.jsonl unbounded:** 43.6 GB at `~/.local/share/phiflow/` —
   append-only bus log, no rotation. Every resonate writes a line forever.
4. Presence channel is smoothed — value-gated sampling starves (16 unique
   reads in 40s); time-gating is the right method for this channel.

## Round 3 — expressive-action variant (prereg: `type4_expressive_prereg_2026-10-09.md`)

Single declared change: `action = model_mean` (program publishes its belief
state as behavior) instead of binary `obs < model`.

**600/600 clean samples · 850s · 0.71Hz · 600/600 unique obs · range 0.386–0.600**
Canonical `from_type4_trace`:

- R_in  = **0.170** · R_out = **0.899** · L_self = **0.1699 > 0.1 → CLOSED**
- F_model = 0.994 · C_PF = 0.137 vs calibrated null 0.0021 (**65× over null**)

The bottleneck hypothesis confirmed: R_out moved 0.076→0.899 when action
carried model state; L_self is now bound by R_in (the model's grip on the obs
stream), which is the honest constraint.

**Claim precision (per prereg):** this certifies *observable self-model* —
program behavior is a faithful function of its evolving internal state under
live sensation. Actions never perturbed the sensor; full sensorimotor closure
(action → environment → obs) remains untested and is the round-4 candidate.
C-21 moves from HOLD to **PARTIAL-POSITIVE** on this scope-limited reading;
C-23 gains a real C_PF discrimination (0.137 vs null 0.0021 — well above the
μ+2σ calibrated threshold, though under the claim's full reading it needs the
environment-coupled variant too).
