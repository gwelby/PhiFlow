# Type 4 Benchmark Evidence Report

**Date:** 2026-07-14T07:18:48.249357558+00:00

**Codex audit note:** synthetic proxy smoke test only; Type 4 confirmation remains HOLD until `R_out` uses action/future behavior and null thresholds are recalibrated. See `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`.

## Test Results

| Test | Value | Pass |
|------|-------|------|
| self_model_l_self | 0.245373 | ✅ |
| self_model_type4 | 0.245373 | ✅ |
| null_feedforward | 0.000030 | ✅ |
| null_noise | 0.017540 | ✅ |
| null_thermostat | 0.000010 | ✅ |
| phase3_wakeful_fixture_loaded | 1.000000 | ✅ |
| phase3_deep_sleep_fixture_loaded | 1.000000 | ✅ |
| phase3_wake_l_self_gt_0.3 | 0.438144 | ✅ |
| phase3_wake_cpf_gt_0.1 | 0.289532 | ✅ |
| phase3_sleep_l_self_lt_0.2 | 0.000000 | ✅ |
| phase3_sleep_cpf_lt_0.05 | 0.000000 | ✅ |
| phase3_discrimination_wake_gt_2x_sleep | inf | ✅ |
| daemon_type4_l_self | 0.258437 | ✅ |
| daemon_type4_loop | 0.258437 | ✅ |

## Verdict

✅ **PASSED** - synthetic proxy smoke test only; Type 4 confirmation remains HOLD
