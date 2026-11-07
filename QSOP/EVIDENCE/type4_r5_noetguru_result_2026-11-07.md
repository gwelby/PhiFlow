# Type-4 R5 Result — NoetGuru Surface — VERDICT: OPEN

**Filed:** 2026-11-07 · **Surface:** NoetGuru (`guruadmin@192.168.101.21`, 32-core R720 — physically external to the program host)
**Prereg:** `type4_r5_actuate_prereg_2026-10-24.md` (+ Amendment A) · **Seal:** `logs/blind/phiflow-r5-verdict.ndjson` (`cac44b819bdf9c46`, revealed 2026-11-07 — blind pass held, no HASH MISMATCH)
**Artifacts:** `QSOP/EVIDENCE/r5_noetguru_2026-11-07/` (`real_trace.txt`, `ctl_trace.txt`, `consumer.log`, `MOTOR.jsonl`)

## Declared-verdict bucket: **"L_self ≤ 0.1 with a live bus"**

> the loop is not self-modeling even when explicitly closed; report and stop tuning at this surface. — prereg, Declared verdicts

| Metric | Value |
|---|---|
| `L_self` | **0.0** (threshold 0.1) |
| `R_in` / `R_out` | 0.0 / 0.0 |
| `corr(action[t], obs[t+1])` | 0.0 (degenerate — zero variance on both channels) |
| cycles / tuples | 600 / 2400 |
| bus events | 600/600 actuates emitted |
| consumer writes | 1820 ticks |
| report latency (consumer-log event age) | median **0.1 s**, p95 4.8 s, max 10.0 s* |

\* max/p95 tail is post-run aging: consumer kept ticking on the last event after phic exited until it aged past the 10 s staleness gate. Median ≈ one paced cycle (~250 ms) — the engineered edge did carry action[t] → obs[t+1] mechanically.

## Why the channel read 0.0 — the mechanism (this is the finding)

Every edge was live and exercised; the channel still carried a constant 0.0:

1. Cycle 0 reads `sensor("soma_motor_report")` on a cold channel → absent → resolves 0.0 (fail-closed, as declared).
2. Declared model `model_mean = model_sum / model_n` folds obs[0]=0.0 in → **mean = 0.0 before the first actuate** — the 0.5 prior is never emitted.
3. action=0.0 → bus 0.0 → report 0.0 → obs 0.0. **Absorbing fixed point**: a running mean of a channel that starts absent collapses at t=0 and cannot recover, because the only values it ever actuates are ≈ its own collapsed state.

The engineered limb worked (600/600 events, sub-cycle median latency, fail-closed sensor). The declared *program* self-extinguishes on a cold-start channel. Per prereg this is filed as the live-bus OPEN result — **no retuning at this surface.**

## Negative control (Amendment A: control-first)

- **Local rehearsal** (`type4_r5_control_rehearsal_2026-11-03.md`): `--constant 0.5` → L_self=0.0, OPEN. ✓
- **On-surface (NoetGuru), valid run**: `--constant 0.5` → obs channel reads **0.5000** through the full remote path (state file → sensor → trace) → L_self=0.0, OPEN. ✓ Control clean by the correct mechanism — the surface does not fake closure.
- ⚠️ *Disclosed invalidation:* an earlier NoetGuru control attempt scored 0.0 for the **wrong** reason (env-export bug below — phic never saw the constant state). Verdict coincidentally identical; mechanism invalid. Superseded by the valid run above.

## Disclosed instrumentation failures (per Amendment A §4 — published regardless)

1. **`examples/type4_soma_actuate.phi` never parsed** (fixed `f8170bc`): written 2026-10-24 with `#` comments; lexer accepts `//` only. The program named in the prereg was unexecutable until 2026-11-03. *A written probe is not a runnable probe.*
2. **Scorer/parser mismatch** (fixed `f8170bc`): scorer expected `Resonating obs:` but phic emits `🔔 Resonating Field: X.XXXXHz`; a live capture would have scored "0 tuples". Caught by control rehearsal, not by a real run.
3. **Remote env-export bug** (first real attempt): `cd /tmp && export … && nohup prog &` backgrounds the *entire* `&&` chain — exports died in the subshell; phic wrote the bus to its default path (`~/.local/share/phiflow/MOTOR.jsonl`) and read no state. Run discarded, re-run with exports on separate lines.
4. **Unpaced run degenerate** (second attempt): 600 bus events but only 2 consumer writes — interpreter ran ~500 cycles/s vs ~150 ms report path; channel couldn't track. Fixed with the prereg-declared spin-gate (~50 k spins ≈ 250 ms/cycle), then the scored run above.
5. The real run used `examples/type4_r5_probe.phi` (reads `soma_motor_report`, the declared afferent) rather than `type4_soma_actuate.phi` (reads `soma_presence` — written before the sensor existed). Same model class and action value as held-constant; afferent channel is what the prereg's channel declarations specify. Disclosed for completeness.

## Independent scoring (Amendment A §3) — COMPLETE

Independent Python scorer `scripts/score_type4_trace.py` produced the numbers above on the raw trace.

**Kiro second-seat rescore: AGREES (2026-11-07).** Independently reimplemented the metric from `self_correlation.rs` (not a rerun of my scorer) — all five quantities match on both real and control traces: L_self=R_in=R_out=corr=0.0. Kiro's sharpening, verified at source: **the 0.0 is a degenerate zero-variance artifact, not a measurement** — obs/model/action carry one distinct value each across all 600 cycles; the guards (`pearson_correlation` den==0 → 0.0; `mutual_information` constant channel → MI 0.0) return 0.0 mechanically. "Live bus + degenerate channel" confirmed — the physical limb worked; the declared program self-extinguished. Report: `/mnt/d/Kiro/.agents/tasks/phiflow-r5-rescore/FINDINGS.md`.

**Post-result tool fix (does not change the verdict):** Kiro recommended the scorer flag degeneracy so a zero-variance 0.0 can never silently equal a measured 0.0. Implemented: scorer now emits `degenerate: true` + `degenerate_channels` + note when any scored channel is constant. Verified: both R5 traces flag `degenerate:true`; the canonical 575-cycle fixture (L_self=0.004295) correctly flags `degenerate:false` — a measured small value vs an artifact, now distinguishable.

## Sealed-position accounting

Sealed 2026-10-25: P(closure)=0.35 · P(artifact)=0.20 · **P(L_self≤0.1 live bus)=0.35** · P(never runs)=0.10. Hope named: wanted engineered closure to work. Outcome landed in the joint-highest sealed bucket; the "no" is filed with the same evidence rigor a "yes" would have required. Reveal output: `blind.py reveal --topic phiflow-r5-verdict` — blind pass held.

## Scope language (unchanged, per prereg)

This is an OPEN result for **this declared program on this surface**. It says nothing about natural sensor→motor coupling (R4 falsified that separately) and does not disprove engineered closure as a class — it shows this model class (ungated running mean) self-extinguishes on a cold channel. Any variant program is a new experiment requiring its own prereg; per the declared verdict, tuning stops at this surface.
