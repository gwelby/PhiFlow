# ENGINEERED-SIGNAL Lane — Result

**Label: ENGINEERED-SIGNAL.** Nothing in this document is natural-SOMA
evidence. It must never be pooled with natural-trace evidence.

- **Date**: 2026-11-07
- **Surface**: NoetGuru (`guruadmin@192.168.101.21`, fabric path) — physically
  external to the development host
- **Prereg**: `QSOP/EVIDENCE/engineered_signal_lane_prereg_2026-11-07.md`
- **Seal**: `fbe51f9bd16281f1` — revealed post-run; blind pass held,
  positions diverge honestly (sealed: P(usable non-degenerate live trace
  first valid run)=0.6 — the bucket the outcome landed in; named hope:
  "the lane works — C-21/C-23 unblocked without waiting on a dead producer")
- **Artifacts**: `QSOP/EVIDENCE/esig_noetguru_2026-11-07/`

## What ran

Producer `scripts/soma_signal_producer.py` writes a complete spec-valid
`SomaState` envelope to `SOMA_STATE_PATH` at 4 Hz with declared waveforms
(presence 60 s period, schumann 47 s, peak_dbc 23 s, slope 31 s, jitter
19 s, phase 41 s — independent periods so channels don't move in
lockstep). The state file carries `engineered_signal: true` as a
provenance marker. Trace program `/tmp/esig_trace.phi` (240 cycles,
~250 ms spin-gate, declared running-mean model class, `action =
model_mean`, obs = `soma_presence`). Scored with the current
`scripts/score_type4_trace.py` (post-`4101dcf`, degeneracy-flagged).

## Results

| Run | degenerate | r_in | r_out | L_self | corr(a,o') | Verdict |
|---|---|---|---|---|---|---|
| Control `--constant 0.5` | **true** (obs+model+action) | — | — | 0.0 | — | OPEN (artifact, as required) |
| Real `--wave` | **false** | 0.3167 | 0.9133 | **0.3167** | 0.2768 | **CLOSED** (> 0.1) |

- Control behaved exactly as the prereg demanded: a constant signal
  produced a `degenerate:true` zero-variance artifact — the instrument
  refused to call a constant a signal. The lane could have vetoed itself;
  it didn't need to.
- The real run produced the **first non-degenerate live Type-4 trace on a
  physically external surface**. r_in = 0.32 means the running-mean model
  genuinely tracks the varying obs through the real `sensor()` path —
  the exact edge that collapsed to an absorbing zero in R5 is alive when
  the channel actually varies.
- Daemon check (T-015 mechanism): `phic --daemon soma_reality_bridge.phi`
  under the wave ticked `Witness [bridge]: Coherence` values 0.5374 →
  0.5109 and falling — the bridge coherence responds to the engineered
  presence channel in real time. Remote binary predates
  `--daemon-state-path`; used `--daemon` + env/default path.

## What this proves

- The full pipeline — producer → state envelope → `sensor()` → daemon/
  probe → trace → scorer — is LIVE end-to-end on an external surface.
- C-21/C-23 now have a non-degenerate trace source for F_model
  calibration **without waiting on AntiGravity's dead natural producer** —
  at engineered evidentiary weight.
- The R5 collapse was the program's absorbing-state bug, not a dead
  surface or dead sensor path — same surface, same paths, varying channel
  → living trace.

## What this does NOT prove

- **Nothing about nature.** Every sensor value here was computed by a
  declared sinusoid. Physical presence did not touch this run.
- The natural coupling question R4 falsified stays falsified.
- T-015's "physical presence stabilizes Council focus" is NOT verified —
  the 15-min daemon run here exercises the *mechanism* (coherence delta
  tracking a live channel) under an engineered signal. The natural
  version still requires AntiGravity's producer.
- r_out = 0.91 is high partly by construction (`action = model_mean`,
  same signal) — consistent with prior lanes, not a new finding.

## Instrumentation failures disclosed

1. Remote `phic` predates `--daemon-state-path` — flag rejected; env
   var/default path used instead. Kit updated to flag-version-check.
2. First bridge invocation (CLI non-daemon mode) compiled then waited —
   `intention/stream` programs need `--daemon`. Recorded in the kit.
3. `pgrep -f` against the ssh wrapper matches itself — status checks can
   read false-positive "RUNNING". Kit updated.
4. Remote scorer copy predated the degeneracy flag — re-scored locally
   with the current scorer. Kit now says: re-deploy the scorer per run.

## Follow-on (not done here)

- T-015 natural: still gated on AntiGravity's producer. This lane supplies
  a separately-labeled mechanism diagnostic at most.
- C-21/C-23 F_model calibration pass against this trace source — next lane
  under its own scope; this lane delivered the trace, not the
  calibration verdict.
