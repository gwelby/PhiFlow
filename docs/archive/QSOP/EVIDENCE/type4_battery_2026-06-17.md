# Type 4 Benchmark Evidence Report

**Date:** 2026-06-17T20:08:07.047035100+00:00

**Codex audit note:** synthetic proxy smoke test only; Type 4 confirmation remains HOLD until `R_out` uses action/future behavior and null thresholds are recalibrated. See `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`.

## Test Results

| Test | Value | Pass |
|------|-------|------|
| self_model_l_self | 0.245373 | ✅ |
| self_model_type4 | 0.245373 | ✅ |
| null_feedforward | 0.000000 | ✅ |
| null_noise | 0.000000 | ✅ |
| null_thermostat | 0.000000 | ✅ |
| phase3_soma_fixtures_available | 0.000000 | ❌ |
| daemon_type4_l_self | 0.258437 | ✅ |
| daemon_type4_loop | 0.258437 | ✅ |

## Notes

- Phase 3 failed: SOMA fixtures not available — skip is not a pass

## Verdict

❌ **FAILED** - SOMA/state-discrimination evidence is missing; Type 4 observer status is not confirmed. Synthetic proxy smoke tests pass, but canonical Type 4 confirmation remains HOLD.
