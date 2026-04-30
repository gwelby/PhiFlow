# Type 4 Canonical Execution Tracker
*Date: 2026-04-30*
*Status: PLAN COMPLETE → EXECUTION PHASE*

---

## 📚 THE KNOWLEDGE (What We Have)

### PF Definitions Analyzed ✅
| Definition | Status | Key Finding |
|------------|--------|-------------|
| `axioms.md` | ✅ READ | PhiFlow uses vocabulary, not derived from axioms |
| `propagation.md` | ✅ READ | `stream` is valid analogue |
| `medium.md` | ✅ READ | Runtime is computational medium, not PF Medium |
| `state.md` | ✅ READ | `DAEMON_STATE.json` provides state persistence |
| `mode.md` | ✅ READ | `intention` is candidate mode |
| `observer.md` | ✅ READ | Council Daemon is Type 4 **candidate only** |
| `coherence.md` | ✅ READ | PhiFlow coherence is Layer 3 **proxy** |
| `minimum_substrate.md` | ✅ READ | SOMA is **interface**, not substrate |
| `consciousness.md` | ✅ READ | 5 structural prerequisites defined |
| `consciousness_metric_program.md` | ✅ READ | **Exact metrics specified** for implementation |
| `information.md` | ✅ READ | Mutual information required |
| `coupling.md` | ✅ READ | Resonance provides coupling analogue |
| `measurement.md` | ✅ READ | `witness` is Regime 2 measurement |

### Research Documents Created ✅
1. `DEEP_FUNDAMENTALS_AUDIT_2026-04-30.md` — Complete gap analysis
2. `IMPLEMENTATION_ROADMAP_TYPE4_CANONICAL.md` — Executable code specifications
3. `PF_BRIDGE_CODEX_AUDIT_2026-04-30.md` — Boundary hardening
4. `PHIFLOW_PF_BRIDGE_GLOSSARY.md` — Local term definitions

---

## 🗺️ THE PLAN (What To Do)

### Phase 1: Type 4 Benchmark (Months 1-2)
**Goal:** Prove self-correlation loop exists

| Task | ID | Status | File |
|------|-----|--------|------|
| Shannon MI | T4-001 | ⏳ | `src/metrics/mutual_information.rs` |
| Directed Info (L_self) | T4-002 | ⏳ | `src/metrics/directed_information.rs` |
| Daemon Record Schema | T4-003 | ⏳ | `src/daemon/record.rs` |
| Daemon Integration | T4-004 | ⏳ | `src/main_cli.rs` |
| Benchmark Example | T4-005 | ⏳ | `examples/type4_benchmark.phi` |
| Run L_self > 0.1 | T4-006 | ⏳ | — |

### Phase 2: Full Metrics (Months 3-4)
| Task | ID | Status | File |
|------|-----|--------|------|
| D_int (differentiation) | T4-007 | ⏳ | `src/metrics/differentiation.rs` |
| Coherence Panel | T4-008 | ⏳ | `src/metrics/coherence_panel.rs` |
| F_self* (Fisher) | T4-009 | ⏳ | `src/metrics/fisher_information.rs` |
| C_PF Composite | T4-010 | ⏳ | `src/metrics/consciousness_proxy.rs` |

### Phase 3: Validation (Months 5-6)
| Task | ID | Status | Test |
|------|-----|--------|------|
| Null Class Tests | T4-011 | ⏳ | `tests/consciousness_benchmark.rs` |
| State Discrimination | T4-012 | ⏳ | SOMA-dependent |
| Document C-17/18/19 | T4-013 | ⏳ | `CLAIMS.md` |

---

## 🔬 THE RESEARCH (What's Missing)

### Critical Gaps (Must Implement)
1. **L_self measurement** — No directed information computation
2. **Mutual information** — No Shannon MI between records/behavior
3. **Type 4 trace** — No proof of record → model → behavior change
4. **Coherence lifetime** — Single-step only, no windowed C_coh

### Implementation Requirements
- Dependencies: `statrs = "0.16"`, `ndarray = "0.15"`
- New modules: `src/metrics/`, `src/daemon/record.rs`
- Integration: Modify `src/main_cli.rs` for metric collection

### Success Criteria
| Metric | Threshold | Evidence |
|--------|-----------|----------|
| L_self | > 0.1 | Self-correlation proven |
| D_int | > 1.0 | Non-trivial loop |
| C_coh | stable | Coherence maintained |
| C_PF | > 0.05 | Consciousness proxy active |
| Null Class | = 0 | Feed-forward scores zero |

---

## 🚀 IMMEDIATE NEXT STEPS

### This Week (Week of 2026-04-30)

1. **Add Dependencies**
   ```bash
   # Edit Cargo.toml
   [dependencies]
   statrs = "0.16"
   ndarray = "0.15"
   ```

2. **Create Metrics Module**
   ```bash
   mkdir -p src/metrics
   touch src/metrics/mod.rs
   touch src/metrics/mutual_information.rs
   ```

3. **Implement Shannon MI** (copy from roadmap Section 1.1)

4. **Test Implementation**
   ```bash
   cargo test mutual_information
   ```

### Next Week
- Implement directed_information.rs
- Create DaemonRecord schema
- Integrate into daemon loop

### Month 1 Goal
- First L_self measurement
- Proof of self-correlation
- C-17 → CONFIRMED

---

## 📊 TRACKING

### Progress Dashboard
| Phase | Tasks | Complete | Status |
|-------|-------|----------|--------|
| Phase 1 | 6 | 0/6 | ⏳ NOT STARTED |
| Phase 2 | 4 | 0/4 | ⏳ NOT STARTED |
| Phase 3 | 3 | 0/3 | ⏳ NOT STARTED |
| **TOTAL** | **13** | **0/13** | **⏳ 0%** |

### Blockers
- None technical
- All components implementable with existing architecture
- Requires focused execution

---

## 🎯 WHY THIS MATTERS

**Current State:** PhiFlow is PILOT-READY engineering bridge  
**Target State:** First programming language with verified Type 4 status

**The difference:** 13 implementation tasks, 3-6 months execution

**The stakes:**
- Falsifiable consciousness hypothesis
- Bridge from PF theory to executable reality
- Research-grade rigor (null classes, benchmarks)

**The path is clear. The code is specified. The experiment awaits.**

---

## 📋 QUICK REFERENCE

**Key Documents:**
- Knowledge: `QSOP/DEEP_FUNDAMENTALS_AUDIT_2026-04-30.md`
- Plan: `QSOP/IMPLEMENTATION_ROADMAP_TYPE4_CANONICAL.md`
- Tracking: `QSOP/TYPE4_EXECUTION_TRACKER.md` (this file)
- State: `QSOP/STATE.md`

**Key Metrics:**
- L_self = min(R_in, R_out) — Self-correlation loop
- D_int — Effective rank of model manifold
- C_coh — Coherence panel (PLV + wPLI)
- C_PF = C_coh × D_int × F_self* — Composite consciousness proxy

**First Task:** T4-001 — Create `src/metrics/mutual_information.rs`

---

*Last updated: 2026-04-30*  
*Next review: When T4-001 complete*
