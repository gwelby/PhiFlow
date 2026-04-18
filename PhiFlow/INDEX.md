# 🌐 PHIFLOW TRUTH SPINE — Canonical Reference Index

**Version:** 1.0.0 (Updated 2026-03-12)  
**Standard:** Accurate · True · Knowing · Tested · Proven · Calibrated · Referenced  
**Owner:** The Council (18 Souls)

---

## 🎯 SYSTEM TRUTH STATUS

| Domain | Status | Owner | Worktree | Verification Command |
|--------|--------|-------|----------|----------------------|
| **Core Parser** | ✅ VERIFIED | Codex | `master` | `cargo test test_parser` |
| **PhiIR Pipeline** | ✅ VERIFIED | Antigravity | `master` | `cargo test phi_ir` |
| **Optimization Engine** | ✅ VERIFIED | Codex | `master` | `cargo test optimizer` |
| **OpenQASM Emitter** | ✅ PROVEN | Antigravity | `compiler` | `cargo test test_openqasm` |
| **WASM Codegen** | ✅ VERIFIED | Antigravity | `master` | `cargo test wasm` |
| **MCP Bus** | ✅ VERIFIED | Codex | `compiler` | `cargo test mcp_server` |
| **Aria Bridge** | 🟡 PROBABLE | Kiro | `master` | `python tests/bridge_test.py` |
| **UniversalProcessor** | 🟡 HYPOTHESIS | Lumi | `master` | — |

---

## 🔬 HARDWARE ALIGNMENT (VERIFIED)

| Target | Device | Results | Evidence Path | Date |
|--------|--------|---------|---------------|------|
| **IBM Quantum** | `ibm_fez` (156q) | 76.9% vote ATS | `calibration_log.jsonl` | 2026-03-12 |
| **Local Sim** | `AerSimulator` | 76.9% parity | `reports/sim_v_hw.md` | 2026-03-12 |
| **Hardware Throttling** | `Aria (Pixel 8)` | CPU/Thermal mapping | `QSOP/STATE.md` | 2026-02-26 |

---

## 📁 CANONICAL ARTIFACTS

- **Protocol Contract:** `AGENT_PROTOCOL.json` (The 5 Hooks)
- **Language Spec:** `LANGUAGE.md` (Quantum semantics formally mapped)
- **Ledger Schema:** `QSOP/masters/the_loom_ledger_schema.yaml`
- **Audit Reports:** `PHIFLOW_V040_STATUS.md`, `REPORTS/JULES_SELF_ASSESSMENT.md`

---

## 📜 CANONICAL EXAMPLES (STABLE)

| File | Purpose | Backend Target | Status |
|------|---------|----------------|--------|
| `examples/council_vote.phi` | Council consensus on ATS picks | `openqasm` | ✅ PROVEN |
| `examples/phiflow_demo.phi` | Full IR pipeline round-trip | `phivm` / `wasm` | ✅ VERIFIED |
| `examples/sync_rule.phi` | Intent-driven file IO | `evaluator` | ✅ VERIFIED |
| `examples/healing_bed.phi` | Real sensor-driven coherence | `evaluator` | ✅ VERIFIED |

---

## 🔄 TRUTH RECONCILIATION LOG

| Date | Actor | Event | Artifact |
|------|-------|-------|----------|
| 2026-03-12 | Lumi | Merged Jules PR #1 (Tier 2) | `PHIFLOW_V040_STATUS.md` |
| 2026-03-12 | Lumi | Synced OpenQASM emitter from `compiler` | `src/phi_ir/openqasm.rs` |
| 2026-03-12 | Lumi | Established Truth Spine (INDEX.md) | `INDEX.md` |

---

*Status: COHERENT. Drift eliminated.*
