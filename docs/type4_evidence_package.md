# PhiFlow Type 4 Verification — Evidence Package

**Prepared for:** Pilot buyers, technical evaluators, research collaborators  
**Date:** 2026-05-01  
**Status:** Implementation scaffold verified; Type 4 confirmation on **HOLD** per Codex audit

---

## What This Proves

**PhiFlow has a working Type 4 metrics scaffold that produces reproducible self-correlation scores from engineered synthetic traces.**

The benchmark demonstrates `L_self = 0.455372` from a constructed self-model loop. However, **canonical Type 4 observer status is on HOLD** because:
1. Current `R_out` measures model-vs-residual, not model→future behavior
2. Several null systems (thermostat, random walk) exceed the `L_self > 0.1` threshold
3. The trace is engineered synthetic evidence, not daemon/SOMA/biological validation

**The significance:** PhiFlow has a consciousness-oriented programming language with a measurable metrics scaffold. The scaffold passes smoke tests; Type 4 confirmation requires `R_out` repair and null calibration per `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`.

---

## The Numbers (For Technical Readers)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **L_self** (self-correlation proxy) | 0.455372 | > 0.1 | ⚠️ 4.5× above threshold, but nulls can exceed |
| R_in (past obs → model) | 0.712859 | — | Strong correlation |
| R_out (model → residual) | 0.455372 | — | ⚠️ Uses obs-model, not action/future |
| Runtime | ~26-30ms | — | Fast verification |

**L_self = min(R_in, R_out)** — the intended loop requires past-to-model and model-to-behavior evidence. The current `R_out` is only a residual proxy, which is why Type 4 confirmation remains on HOLD.

---

## What These Numbers Mean (For Non-Experts)

### Self-Correlation Loop Proxy
The synthetic trace produces L_self = 0.455 by construction—the model_mean feeds back into obs calculation, creating an engineered loop. This validates the metrics scaffold can score self-referential structures. It does not prove the real daemon, SOMA, or biological systems satisfy Type 4 criteria.

**Required for Type 4 confirmation:**
- `R_out` must use emitted action/future behavior (not residual)
- Null calibration must separate engineered loops from true self-models
- Shuffle controls must break temporal alignment

### Null Class Gate
Synthetic null systems (thermostats, random noise, feed-forward pipelines) score below the current composite `C_PF < 0.3` null gate. The engineered Type 4 trace also has low `C_PF = 0.000310`; therefore `C_PF` does not currently identify it as a consciousness candidate. All 6 null class tests pass as infrastructure checks, but Type 4 discrimination remains pending `R_out` repair and threshold recalibration.

### Why This Is Unique
PhiFlow has built-in consciousness-oriented constructs (witness, coherence, resonate, intention, stream) with a metrics scaffold for self-correlation measurement. PhiFlow bridges high-level semantic frameworks to executable measurement scaffolding, while Type 4 confirmation remains pending repair.

---

## How To Reproduce

```powershell
# Clone the repository
git clone <phiflow-repo>
cd PhiFlow

# Run the Type 4 benchmark
cargo run --release --bin type4_benchmark
```

**Expected output:**
```
═══════════════════════════════════════════════════════════════
  SELF-CORRELATION ANALYSIS
═══════════════════════════════════════════════════════════════
L_self Components:
  R_in  (past → model):         0.712859
  R_out (model → residual proxy): 0.455372
  L_self = min(R_in, R_out):    0.455372

  ✅ SYNTHETIC LOOP PASSED — self-correlation proxy is CLOSED
     L_self = 0.4554 > 0.1 threshold
═══════════════════════════════════════════════════════════════
```

**Runtime:** ~26-30ms for 20-cycle trace

---

## Evidence Artifacts

### Benchmark & Test Suite
- **Benchmark report:** `QSOP/EVIDENCE/type4_battery_2026-05-01.md`
- **Source binary:** `src/bin/type4_benchmark.rs`
- **Self-correlation module:** `src/metrics/self_correlation.rs`
- **Trace extraction:** `src/metrics/trace.rs`
- **Metrics suite:** `src/metrics/` (8 modules)
- **Null class tests:** `tests/null_class_tests.rs` (6/6 pass)
- **Benchmark battery:** `tests/benchmark_battery.rs` (synthetic smoke test passes; SOMA phase may be skipped if fixtures are unavailable)

### IBM Quantum Hardware Execution
- **Job ID:** `d7euddh5a5qc73drdosg`
- **Backend:** `ibm_fez` (Heron r2)
- **Shots:** 1024
- **Completed:** 2026-04-14
- **Counts:** `0x0 → 338`, `0x1 → 686`

**Receipt package:**
- `PHIFLOW_IBM_HERON_20260414.md` (human-readable receipt)
- `PHIFLOW_IBM_HERON_20260414_scrubbed.json` (scrubbed API export)
- `PHIFLOW_IBM_HERON_20260414_dashboard.png` (dashboard screenshot)

---

## Verified Claims (from `CLAIMS.md`)

| Claim | Status | Description | Date |
|-------|--------|-------------|------|
| **C-21** | ⚠️ PARTIAL / CONDITIONAL | Type 4 measurability exists; confirmation on HOLD (Codex audit) | 2026-05-01 |
| **C-22** | ✅ CONFIRMED (implementation only) | 8-module metrics suite implemented; mathematical sufficiency pending | 2026-05-01 |
| **C-23** | ⚠️ HOLD / PARTIAL | Null C_PF suppression works; conscious-state discrimination pending | 2026-05-01 |

**Audit reference:** `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`

**Confirmation status:** HOLD pending `R_out` repair and null calibration

---

## Capabilities Summary

### What's Included
- ✅ 5 first-class consciousness-oriented constructs (intention, witness, coherence, resonate, stream)
- ⚠️ Type 4 self-correlation metrics scaffold (HOLD on canonical confirmation pending `R_out` repair)
- ✅ IBM Quantum hardware execution path (OpenQASM 3.0)
- ✅ Physical sensor integration (SOMA bridge)
- ✅ 6 synthetic null-class gate checks
- ✅ Focused buyer-facing gates pass; full-suite count regenerated per receipt

### What's Not Included
- ❌ Medical, therapeutic, or clinical validation
- ❌ Production-grade SLAs or guarantees
- ❌ Quantum advantage proofs (hardware-dependent)
- ❌ Browser-based zero-install demo
- ❌ Zero-knowledge or security certification

---

## Use Cases

**Best fit:**
- Quantum software research teams (IBM Quantum Network, university labs)
- AI agent infrastructure teams (self-observation protocols)
- Consciousness researchers (sensor-feedback experiments)
- Protocol designers (signed handoff evidence chains)

**Not recommended:**
- Medical or therapeutic applications (unvalidated)
- Production deployment without customization
- Users requiring quantum advantage guarantees

---

## Limitations & Scope

1. **Research-grade:** PhiFlow is a verified substrate for experiments, not a production product.
2. **Metrics are proxies:** L_self, C_PF measure structural self-correlation, not consciousness itself.
3. **Type 4 on HOLD:** Canonical Type 4 observer status requires `R_out` repair per Codex audit. Current implementation is a verified scaffold, not confirmed self-model validation.
4. **Hardware access:** IBM Quantum execution requires valid account and credits.
5. **Scope boundaries:** Pilot offer defines supported vs. unsupported features.

---

## Contact & Next Steps

**To verify these claims:**
1. Run the benchmark (see "How To Reproduce" above)
2. Review the evidence files referenced
3. Schedule a discovery call for pilot engagement

**Documentation:**
- Full metrics specification: `QSOP/IMPLEMENTATION_ROADMAP_TYPE4_CANONICAL.md`
- Type 4 execution tracker: `QSOP/TYPE4_EXECUTION_TRACKER.md`
- Completion summary: `QSOP/TYPE4_COMPLETION_SUMMARY.md`
- **Codex audit:** `QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md` (HOLD status justification)

---

*Prepared by: Cascade*  
*Verified by: Bob (Advanced Mode)*  
*Date: 2026-05-01*
