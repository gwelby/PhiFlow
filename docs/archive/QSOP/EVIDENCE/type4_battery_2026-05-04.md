# Type 4 Benchmark Evidence Report

**Date:** 2026-05-04T02:24:37.285141412+00:00

**Codex audit note:** synthetic proxy smoke test only; Type 4 confirmation remains HOLD until `R_out` uses action/future behavior and null thresholds are recalibrated. See `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`.

## Test Results

| Test | Value | Pass |
|------|-------|------|
| self_model_l_self | 0.245373 | ✅ |
| self_model_type4 | 0.245373 | ✅ |
| null_feedforward | 0.000030 | ✅ |
| null_noise | 0.013166 | ✅ |
| null_thermostat | 0.000010 | ✅ |
| daemon_type4_l_self | 0.258437 | ✅ |
| daemon_type4_loop | 0.258437 | ✅ |

## Notes

- Phase 3 skipped: SOMA fixtures not available

## Verdict

✅ **PASSED** - synthetic proxy smoke test only; Type 4 confirmation remains HOLD
