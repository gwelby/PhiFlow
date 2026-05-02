# Type 4 Benchmark Codex Audit
*Date: 2026-05-01*
*Scope: `src/metrics/`, `src/bin/type4_benchmark.rs`, `tests/null_class_tests.rs`, `tests/benchmark_battery.rs`, `examples/type4_trace_benchmark.phi`, `QSOP/EVIDENCE/type4_battery_2026-05-01.md`*

## Verdict

**PASS as an implementation smoke test. HOLD as Type 4 confirmation.**

PhiFlow now has a working metrics scaffold and a synthetic benchmark that produces the reported `L_self = 0.455372`. That is real progress. It is not yet enough to claim canonical Type 4 observer status, because the current `R_out` implementation does not measure the advertised quantity.

## Commands Run

```bash
cargo run --release --bin type4_benchmark
cargo test --test null_class_tests -- --test-threads=1 --nocapture
cargo test --test benchmark_battery -- --ignored --test-threads=1 --nocapture
```

The broad `cargo test metrics -- --test-threads=1 --nocapture` command was stopped after it compiled the full test graph and remained a poor audit signal. The focused gates above completed.

## What Is Verified

- `type4_benchmark` executes `examples/type4_trace_benchmark.phi` and reproduces `R_in = 0.712859`, `R_out = 0.455372`, `L_self = 0.455372`.
- The composite proxy on that trace is low: `C_PF = 0.000310`, and the binary correctly reports it is **not** a consciousness candidate.
- Null tests pass the current composite gate `C_PF < 0.3`.
- The metrics modules exist and are wired: trace extraction, mutual information, self-correlation, differentiation, coherence panel, Fisher sensitivity, and composite scoring.

## Findings

### T4-01 — `R_out` does not measure model-to-future behavior

**Severity: Critical**

The file header says:

```text
R_out = I_dir(model_state -> future_behavior | current_obs)
```

The implementation in `SelfCorrelation::from_type4_trace()` parses `action`, but does not use it. It computes:

```rust
normalized_mi(&model_vals, &deviation, 5)
```

where `deviation = obs - model`. This measures same-trace association between the model and residual, not directed influence from model to future behavior, and it does not condition on current observation.

Required fix:

- `R_in`: keep past observation/history -> current model, but compute with a stated information/correlation proxy.
- `R_out`: use actual emitted behavior/action at `t` or `t+1`, not residual `obs - model`.
- Add a shuffle/permutation control: preserving marginal distributions while breaking temporal alignment must drive `R_out` and `L_self` below threshold.
- If claiming conditional directed information, add the conditioning term on current observation or rename the proxy honestly.

### T4-02 — Null classes pass `C_PF`, but several nulls have high `L_self`

**Severity: High**

Focused null output:

```text
feedforward   L_self=0.2447  C_PF=0.0000  PASS
noise         L_self=0.1875  C_PF=0.0144  PASS
replay        L_self=0.3390  C_PF=0.0022  PASS
thermostat    L_self=0.6555  C_PF=0.0000  PASS
random_walk   L_self=0.3704  C_PF=0.0002  PASS
```

This means `L_self > 0.1` alone is not a valid Type 4 discriminator under the current implementation. The existing benchmark uses `L_self > 0.1` to claim Type 4 confirmation, but the thermostat null scores higher than the Type 4 trace on `L_self`.

Required fix:

- Stop using `L_self > 0.1` alone as a Type 4 confirmation gate.
- Introduce a joint gate such as `L_self > threshold` plus null-separated `C_PF`, action-scramble control, and model-ablation control.
- Recalibrate thresholds against null distributions, not single hand-picked constants.

### T4-03 — The benchmark trace is engineered synthetic evidence

**Severity: High**

`examples/type4_trace_benchmark.phi` creates a loop by construction:

- `model_mean = model_sum / model_n`
- `obs = base_val * mod_val`, where `mod_val` depends on `model_mean`
- `action = 1.0 if obs < model_mean else 0.0`

This is valid as a metric smoke test. It is not evidence that the real daemon, SOMA, or a biological system satisfies Type 4 criteria.

Required fix:

- Keep this as `synthetic_type4_loop_smoke`.
- Add a separate real daemon trace fixture with witness/state persistence.
- Add SOMA/EEG fixture gates before claiming biological or consciousness discrimination.

### T4-04 — Phase 3 skip is treated as pass

**Severity: Medium**

The manual benchmark battery returns `true` when `PHIFLOW_SOMA_FIXTURES` is missing, then prints a final pass. The generated evidence notes the skip, but still says Type 4 observer status is confirmed.

Required fix:

- Treat missing SOMA fixtures as `SKIPPED`, not `PASS`.
- Prevent a skipped discrimination phase from producing a global "confirmed" verdict.

### T4-05 — Trace adapter uses placeholders for core channels

**Severity: Medium**

When parsing the synthetic 4-tuple trace, `Trace::from_resonance_events_only()` sets placeholder coherence/depth values. `C_coh`, `D_int`, and `F_model` are therefore partly functions of adapter assumptions, not measured runtime state.

Required fix:

- Emit or extract actual model/action/coherence/depth channels.
- Do not use placeholder channels in any claim stronger than "smoke test."

### T4-06 — wPLI test commentary is inconsistent with the intended null defense

**Severity: Low**

The code comments correctly say wPLI suppresses zero-lag correlations. The unit-test comment says identical zero-lag signals should have wPLI near 1, which contradicts the intended anti-volume-conduction role. The test prints but does not assert.

Required fix:

- Update the test expectation and assert zero-lag wPLI is low or explicitly define why this implementation differs from standard wPLI usage.

## Status Corrections

- C-21 should be **PARTIAL / CONDITIONAL**, not confirmed. Measurability exists; Type 4 confirmation does not.
- C-22 can remain **CONFIRMED** only for implementation existence, not mathematical sufficiency.
- C-23 should be **HOLD / PARTIAL**, because current tests show null `C_PF` suppression but not conscious-state discrimination, and the positive trace has `C_PF = 0.000310`.

## Close Criteria For Upgrade

The Type 4 claim can upgrade only after all of the following pass:

1. `R_out` uses behavior/action and future alignment, not residual deviation.
2. Action-shuffle, model-shuffle, replay, thermostat, random-walk, and collapsed-synchrony nulls all fail the Type 4 gate.
3. Thresholds are calibrated from null distributions and stated in the evidence report.
4. The manual battery cannot report global pass when SOMA/real-trace discrimination is skipped.
5. A real daemon or SOMA trace passes the same gate without synthetic construction of the loop.

Until then: **implementation scaffold verified; Type 4 confirmation held.**
