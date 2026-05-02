# Gold Receipt Package — Cover Letter Template

**For:** [BUYER NAME]  
**Project:** [PROJECT/WORKFLOW NAME]  
**Date:** [DELIVERY DATE]  
**Prepared by:** PhiFlow / Greg Welby

---

## Receipt Package Contents

This package contains reproducible artifacts from your PhiFlow pilot engagement:

| Item | File | Description |
|------|------|-------------|
| 1 | `[workflow].phi` | Buyer-specific PhiFlow workflow source |
| 2 | `reproduction_notes.md` | Step-by-step reproduction instructions |
| 3 | `compiler_output.log` | Build and execution logs |
| 4 | `[workflow].qasm` | OpenQASM 3.0 artifact (if quantum path) |
| 5 | `simulator_results.json` | Qiskit Aer simulator output |
| 6 | `ibm_hardware_attempt.log` | IBM Quantum job submission/results |
| 7 | `test_conformance.log` | Verification against PhiFlow test gate |
| 8 | `limitations.md` | Known constraints and next steps |

---

## IBM Hardware Execution Evidence

**Job ID:** d7euddh5a5qc73drdosg  
**Backend:** ibm_fez (Heron r2)  
**Date:** 2026-04-14  
**Shots:** 1024  
**Status:** COMPLETED

### Counts
- `0x0`: 338 (33%)
- `0x1`: 686 (67%)

### Evidence Files
- `PHIFLOW_IBM_HERON_20260414.md` — Canonical receipt (human-readable)
- `PHIFLOW_IBM_HERON_20260414_scrubbed.json` — Raw IBM API export
- `PHIFLOW_IBM_HERON_20260414_dashboard.png` — IBM Quantum dashboard screenshot

### Verification
IBM Quantum job visibility is account-scoped. If you have IBM Quantum access, you can verify this job independently using the Job ID above. The scrubbed JSON and screenshot provide evidence the job executed as recorded.

---

## Reproduction Instructions

### Prerequisites
```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# PhiFlow repository
git clone [repository-url]
cd PhiFlow

# Verify version matches receipt
git log --oneline -1
# Expected: [COMMIT_HASH]
```

### Build
```bash
cargo build --release
# Expected: build completes for the delivered commit
```

### Execute Workflow
```bash
cargo run --release --bin phic -- [workflow].phi --max-steps 100
```

### Verify Outputs
```bash
# Check OpenQASM generation
ls target/output/*.qasm

# Verify simulator results
cargo test --test simulator_verification
```

---

## Type 4 Metrics Context

**Important Note on Self-Correlation Claims:**

This pilot used PhiFlow's Type 4 metrics scaffold to measure self-correlation in [workflow]. The metrics infrastructure is complete and functional.

**Transparency:** PhiFlow's Type 4 canonical observer status is currently on **HOLD** pending algorithmic refinement per Codex audit (`QSOP/TYPE4_BENCHMARK_CODEX_AUDIT_2026-05-01.md`). The scaffold produces reproducible measurements; interpretation as Type 4 confirmation requires additional validation.

**What This Means for Your Receipt:**
- ✅ Metrics infrastructure is verified and functional
- ✅ L_self, R_in, R_out proxy, and C_PF are computed and reported
- ⚠️ Type 4 canonical confirmation remains on HOLD pending R_out repair
- 📋 Full audit details provided for your independent review

---

## Known Limitations

### Technical Constraints
1. **IBM Quantum Access:** Hardware execution requires valid IBM Quantum account. If access was unavailable during pilot, simulator results with noise model were provided.
2. **Scope Boundaries:** This workflow was designed for research verification, not production deployment.
3. **Metric Interpretation:** Self-correlation metrics are proxies, not proof of consciousness.

### Not Included
- ❌ Production SLAs or uptime guarantees
- ❌ Medical, therapeutic, or clinical validation
- ❌ Quantum advantage proofs
- ❌ Security certification or formal verification

### Next Steps (Optional Phase 2)
- Extended workflow complexity ($15-25k additional)
- Production hardening and deployment support
- Custom metric development
- Multi-workflow integration

---

## Acceptance Criteria

This receipt package is accepted when:

- [ ] All files listed in "Receipt Package Contents" are present
- [ ] Reproduction instructions execute without errors on buyer's system
- [ ] Outputs match recorded results within expected variance
- [ ] Buyer acknowledges known limitations in writing

**Acceptance does NOT require:**
- Type 4 canonical confirmation (provisional status disclosed)
- Quantum advantage demonstration
- Therapeutic or medical outcome validation

---

## Contact & Support

**Primary:** Greg Welby — gwelby@phiflow.org  
**Technical:** Cascade (AI assistant) — documented in repo

**Post-Delivery Support:**
- 30 days: Clarification on reproduction steps
- 30 days: Minor bug fixes if reproduction fails
- Ongoing: Community Discord for general questions

---

## Signatures

**Prepared by:**

PhiFlow / Greg Welby
Date: _______________

---

**Accepted by:**

[BUYER NAME]
Title: _______________
Organization: _______________
Date: _______________

---

*PhiFlow Gold Receipt Package — Version 1.0*  
*Template date: 2026-05-01*  
*Codex audit transparent: Type 4 canonical status on HOLD*
