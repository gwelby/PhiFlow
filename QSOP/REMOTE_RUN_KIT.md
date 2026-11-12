# Remote Run Kit — PhiFlow on NoetGuru / 8086K

Reusable pattern for running `phic` experiments on the batch surfaces.
Distilled from R5 (2026-11-07) and the ENGINEERED-SIGNAL lane (same day) —
three instrumentation failures and their fixes are baked in here.

## Lane rules (MACHINES.md — canonical registry; hosts/users/keys live
there, NOT in this repo — the credential guard will block key paths)

- **NoetGuru** (fabric path in MACHINES.md — mgmt path dead):
  batch executor. Inputs copy to it; the job runs there; coordination
  state NEVER lives there. CPU-only `phic` needs no GPU lease.
- **8086K** (inference box, 12 cores / 45 GB / RTX 2060 + GTX 1070,
  dedicated key already installed): GPU jobs go through `gpu_gate.sh` +
  a registered lease + blackboard announce. **Proven `phic` surface
  2026-11-11** (`QSOP/EVIDENCE/esig_8086k_surface_validation_2026-11-11.md`).
  Guest rules: it hosts kobold (1070) and other people's GPU jobs (e.g. a
  hackathon on the 2060) — run `phic` with `CUDA_VISIBLE_DEVICES=""` and
  `nice -n 19`, one job at a time, in `~/phiflow_surface/<date>/` (its
  `/tmp` is RAM-backed tmpfs). Check `uptime` + `nvidia-smi` first.

## Cross-surface comparison: pace by clock, not iterations

Spin-gates (`while gate < 50000`) pace by CPU speed. The same probe ran
240 cycles in ~65 s on NoetGuru and 19 s on 8086K, so it sampled a full
60 s presence wave on one and a monotone quarter on the other (L_self
0.317 vs 0.948 — the second is a trend artifact). Verdicts may transfer;
magnitudes do not, until the probe paces by clock time. Never pipe a
check (`pgrep`) into the same ssh stream as a `tar` artifact pull.

**Measured 2026-11-11:** with `sensor("clock_s")` deadline pacing
(`examples/type4_esig_clock_trace.phi`, `scripts/run_esig_clock.sh`), both
surfaces ran 240 cycles in 60.25 s and agreed to 0.0002 (L_self 0.681).
Use that pattern for any time-sensitive probe. State an L_self magnitude
together with its window **in signal time**, because the value depends on
how much of the signal's period was sampled.

**For new probes (phic built after 2026-11-11):**
- Wait with `sleep(5)` inside the deadline loop instead of spinning. Checked
  locally on the clock probe: the same 60 s run and the same L_self (0.6676
  vs 0.6677), CPU 56 s → 0.8 s. This matters on a shared surface (8086K's
  hackathon GPU host).
- Run with `--strict-sensors`, so a dead producer stops the run instead of
  scoring a channel of 0.0 (the R5 invalid-run scar).
- Run with `--resonance-tsv`. `score_type4_trace.py` then pairs values by their
  `as "step"/"obs"/"model"/"action"` labels at full precision, and a missing
  or extra emission is an error instead of a silent shift. Legacy stdout still
  scores exactly as before.
- Sealed probes (`type4_esig_clock_trace.phi` at `7a9b796`) stay byte-identical;
  apply these in a new probe file under a new prereg.

## Deploy contract

```bash
# KEY/H per MACHINES.md (do not commit key paths — the guard enforces)
scp -i $KEY -o BatchMode=yes \
    target/release/phic \
    scripts/soma_signal_producer.py \
    scripts/soma_motor_report_consumer.py \
    scripts/score_type4_trace.py \
    <program>.phi \
    $H:/tmp/
```

`/tmp/phic` on NoetGuru is an OLDER build — check flags before assuming
(`--daemon-state-path` does not exist there; use `PHIFLOW_DAEMON_STATE_PATH`
env or the `/tmp/phiflow_daemon_state.json` default). Re-deploy the binary
when the run needs a newer feature.

## THE export scar (R5, cost a full invalid run)

`cd /tmp && export SOMA_STATE_PATH=... && nohup prog &` backgrounds the
ENTIRE `&&` chain — exports die with the subshell, the foreground process
silently uses DEFAULT paths. A "control" can still produce the right
verdict for the wrong reason (dead channel reads 0 either way).

Correct pattern — exports on their own lines, then start:

```bash
ssh -i $KEY $H 'export SOMA_STATE_PATH=/tmp/run/state.json
export MOTOR_BUS_PATH=/tmp/run/MOTOR.jsonl
export PHIFLOW_DAEMON_STATE_PATH=/tmp/run/daemon_state.json
mkdir -p /tmp/run
nohup prog > /tmp/run/prog.log 2>&1 &
echo started=$!'
```

## Control-first checklist (sealed-experiment skill, step 3)

1. Producer `--constant V` → run probe → scorer MUST print
   `degenerate: true` on all channels. If it doesn't, the instrument is
   scoring its own machinery — veto the run.
2. Varying run → `degenerate: false` + inspect whether variation is real
   (check raw obs channel, not just L_self).
3. Score with the CURRENT scorer — re-deploy `score_type4_trace.py`; the
   remote copy rots.

## Mechanism checks before scoring (all learned the hard way)

- Bus actually written: `wc -l $MOTOR_BUS_PATH` > 0 before celebrating.
- Consumer writes: `grep -c wrote consumer.log` comparable to bus events
  (pace the probe — unpaced ~500 cyc/s vs ~150 ms report path = starved).
- `pgrep -f "phic /tmp/x"` matches the ssh wrapper itself — use
  `pgrep -f` from a different pattern or check output file growth.
- `/tmp/phic` ENOENT with the file present = cwd/race artifact; use
  absolute paths everywhere.
- Stdout is buffered — a 1-line trace file while running is normal;
  check `ps etime`, not the file.
- Probe files must COMPILE: `#` comments are not comments here (`//` only)
  — run `phic program.phi` once before committing it to a prereg.

## Artifact pull + disclosure

Pull before cleanup: `scp $H:/tmp/run/{trace,state,log,daemon_state}* .`
Stage to `QSOP/EVIDENCE/<lane>_<surface>_<date>/`. `*.log` is gitignored —
commit `.md` evidence + raw traces; disclose every instrumentation
failure in the result doc regardless of which bucket the verdict landed in.

## Producers

- `scripts/soma_signal_producer.py` — ENGINEERED-SIGNAL state writer:
  `--wave` (declared multi-period sinusoids), `--constant V` (negative
  control), `--drift` (presence ramp), `--once` (single write).
  Complete spec-valid SomaState envelope, atomic writes, writes
  `engineered_signal: true` provenance marker into the state itself.
- `scripts/soma_motor_report_consumer.py` — motor-bus → `soma_motor_report`
  (R5 closure channel). `--constant` for its control, `--interval` daemon
  logs per-tick `age` = bus-latency histogram.
