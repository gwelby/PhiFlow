# Type 4 Benchmark Evidence Report

**Date:** 2026-05-01T03:21:56.035992761+00:00

**Codex audit note:** synthetic proxy smoke test only; Type 4 confirmation remains HOLD until `R_out` uses action/future behavior and null thresholds are recalibrated. See `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`.

## Test Results

| Test | Value | Pass |
|------|-------|------|
| self_model_l_self | 0.245373 | ✅ |
| self_model_type4 | 0.245373 | ✅ |
| null_feedforward | 0.000030 | ✅ |
| null_noise | 0.014300 | ✅ |
| null_thermostat | 0.000010 | ✅ |
| daemon_type4_l_self | 0.455372 | ✅ |
| daemon_type4_loop | 0.455372 | ✅ |

## Notes

- Phase 3 skipped: SOMA fixtures not available

## Verdict

✅ **PASSED** - synthetic proxy smoke test only; Type 4 confirmation remains HOLD
