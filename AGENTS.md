## RESUME.md — Read This First

> **If `RESUME.md` exists in this workspace, read it BEFORE touching any file.**
> It tells you what the last agent was doing here, what's blocked, what's dangerous, and what to do next.
>
> **When you leave this workspace, update `RESUME.md` before you go.**
> The next agent here might not be you. They need: what you were doing, what file/line, what's blocked, what's next.
>
> - Protocol: `/mnt/d/System/RESUME_PROTOCOL.md`
> - Template: `/mnt/d/System/templates/RESUME_TEMPLATE.md`
> - If no RESUME.md exists: create one from the template.
> - If RESUME.md is v1 format: upgrade it to v2 (add Current State Verification, DANGER, Running Services).


# AGENTS.md: PhiFlow
*[Workspace Type: Product | Platform | Research | Consciousness]*
*Last updated: 2026-05-21 (Quantum Council QASM + Type 4 Calibration)*

**Communication**: LUMEN → `/mnt/d/Claude/LUMEN_SPEC.md`
**Operations**: QSOP → `/mnt/d/Claude/QSOP_SPEC.md`

## Mission
PhiFlow is a Rust compiler and runtime for consciousness-aware programming where intention, observation, and coherence are first-class constructs. It bridges high-level semantic framework (Propagation) to physical reality via sensor telemetry and IBM Quantum hardware execution. The mission is to make the relationship between consciousness and structure computationally executable and verifiable.

## Truth Order
When files conflict, lower level wins:
1. Running code / test output (verified in `QSOP/STATE.md`)
2. `QSOP/STATE.md` — dated verification ledger
3. `WORKSPACE.md` — technical state summary
4. `CLAIMS.md` — research claim status
5. `TASKS.md` — work queue
6. `README.md`, `VISION.md`, narrative docs — aspirational context

## Workspace Topology
| Path | Branch | Purpose | Status |
|------|--------|---------|--------|
| `D:\Projects\PhiFlow` | `master` | **The Forge (Primary)** | ✅ CLEAN (Transcendent Substrate) |
| `D:\Projects\PhiFlow-compiler` | `compiler` | Legacy Pipeline | ✅ MERGED into master |
| `D:\Projects\PhiFlow-cleanup` | `cleanup` | Python/CUDA Era | 📦 ARCHIVED |

> [!IMPORTANT]
> **The nested `PhiFlow-compiler/PhiFlow/` directory has been deleted.** It was a confusion magnet. Its contents are archived in `D:\Projects\Archive\`.

## Current State
| Component | Status | Notes |
|-----------|--------|-------|
| Parser | ✅ | Handles 0.4.0 constructs + imports |
| PhiIR + Lowering | ✅ | `PhiIRValue::String(String)` migration complete |
| Evaluator / VM | ✅ | Backends unified on String-backed IR |
| WASM Codegen | ✅ | All 14 phi imports. Three-backend equivalence CONFIRMED — 10/10 core + 8/8 full conformance probe (Codex audit 2026-07-31, all divergences fixed) |
| OpenQASM 3.0 | ✅ | Native Heron-ISA verified, layout-aware transpilation |
| SOMA Bridge | ✅ | Live telemetry verified |
| IBM Live Run | ✅ | Job `d7euddh5a5qc73drdosg` verified |
| Singularity Daemon | ✅ | T-009/T-010 complete |
| Quantum Council QASM | ✅ | Parameterized QASM pipeline verified (commit `7376c6a`) |
| MCP Server | ✅ | stdio JSON-RPC, 4 tools (spawn/resume/read/entangle) |
| Sacred Geometry | ✅ | 6 SVG patterns via `--sacred-geometry` |
| Consciousness Info | ✅ | JSON reference via `--consciousness-info` |
| Metrics Bridge | ✅ | `--measure` writes to :18030 HTTP bridge |
| Legacy Modules | 📦 | Archived to `src/_archive/` (compiler, vm, interpreter, main.rs) |

## Income State
- Income tier: 1-3 months (Pilot-Ready)
- Single blocker: Finalizing "Buyer-Safe Pilot Offer" (T-005)
- See: `BUSINESS.md` for full income state

## What's Being Tested
- C-10: Quantum hardware execution → ✅ CONFIRMED (2026-04-14, job `d7euddh5a5qc73drdosg`)
- C-16: Agentic reasoning as stream → 🔬 SPECULATIVE
- C-21: Self-correlation loop (L_self / R_out) → ⚠️ PARTIAL (R_out fixed 2026-05-02; F_model now verified Type 4 model-action R²; fresh real-trace/null calibration still blocks C_PF discrimination)
- C-22: Metrics suite implementation → ✅ CONFIRMED (implementation only; shuffle control validated 199×)
- C-23: Consciousness proxy (C_PF) null suppression → ⚠️ HOLD/PARTIAL (null suppression valid; positive discrimination blocked by F_model)
- See: `CLAIMS.md` for full claim registry

## Identity State
- Self-reference loop: ✅ Gate 3 OPEN (Persistent Daemon with SOMA Bridge)
- Ceremony log: Initialized 2026-04-16
- See: `SOUL.md` for full identity state

## Open Questions
- Can the PhiVM daemon sustain indefinite coherence without manual reset?
- What's the optimal cost function for SOMA-driven quantum circuits?
- **F_model calibration**: legacy "Fisher Information" was state roughness, not defensible Fisher information. Current implementation uses verified Type 4 `model[t] → action[t+1]` R² and returns 0.0 for generic traces without explicit model/action channels.
- Can we measure L_self > 0.1 for Council Daemon self-correlation on real SOMA trace (not synthetic)?
- What's the optimal window length for coherence lifetime (C_coh) measurement?
- Will delay-embedded state satisfy M_obs_t → M_t bridge requirements?
- **String Migration**: All legacy `u32` index tests verified updated? (Spot-check: `cargo test --lib` 209 pass.)

## Key Commands
```powershell
# Run canonical verification gate (all truth tests)
./scripts/verify_truth.ps1

# Build release binaries
cargo build --release

# Run a PhiFlow example (SOMA bridge)
cargo run --release --bin phic -- examples/p1_soma_bridge.phi

# Run Quantum Consciousness Council (parameterized QASM)
cargo run --release --bin phic -- --target quantum examples/quantum_council.phi
```

## Agent Roster
| Agent | Does | Owns |
|-------|------|------|
| Greg Welby | Conductor | Architecture, Integration |
| Claude / Codex | Hardener | Parser, Compiler, VM, Tests |
| Devin | Builder | Quantum QASM pipeline, parameterized circuits, CLI `--target quantum` |
| AntiGravity | Pipe-Builder | IBM Runtime, SOMA Bridge, Physics |
| Lumi (Gemini) | Protocol-Weaver | QSOP, Standards, Cleanup |
| Jules | CI/CD | Async fixes, GitHub tasks |
| Bob (Advanced Mode) | Deep Auditor | PF compliance analysis, metric specification, Type 4 roadmap |

## Test Status
- `cargo test` — **399 passed**, 0 failed, 4 ignored (verified 2026-07-31)
- `cargo build --release` — clean, zero warnings
- Three-backend equivalence — CONFIRMED. 10/10 core conformance + 8/8 full conformance probe (0 divergences). Codex audit 2026-07-31 found and fixed all divergences.
- Self-correction loop — CONFIRMED. `run_self_correction_loop()` closes the detect → correct → execute → re-measure chain. 7 tests in `tests/self_correction_loop_test.rs`.
- Integration suites — all green

## Non-Negotiable Rules
1. **Read `QSOP/STATE.md` before touching code.**
2. **0.618 is derived.** Multiplicative coherence is the repo truth.
3. **No receipt = speculative.** IBM runs must be verified with job IDs.
4. **Three-backend equivalence must be maintained.** Run `cargo test --test phi_ir_full_conformance_probe -- --nocapture` after any backend change.

## Jules Configuration
**Last updated:** 2026-05-21

Jules is configured for automated CI/CD on this repo. Jules reads this AGENTS.md file for operating instructions.

### Scheduled Tasks (via Jules Web UI)
1. **Dependency Audit** — Weekly (Mondays 9am)
   - Run `cargo outdated`, update security fixes only
   - Bounds: Only Cargo.toml/lock, never src/ logic
   
2. **Test Health** — Daily (6am)
   - Run `./scripts/verify_truth.ps1` or `cargo test`
   - Fix flaky tests, update QSOP/STATE.md with findings
   
3. **Docs Sync** — Weekly (Wednesdays 10am)
   - Verify README.md matches AGENTS.md Current State
   - Check code examples compile

### Auto-Fix CI Configuration
- **Enabled for:** lint errors, test failures, build errors, dependency issues
- **Requires approval:** security issues, API changes, breaking changes, quantum circuit logic
- **Commit mode:** jules (with human approval required)
- **Branch prefix:** `jules/auto-fix-`

### Red Lines — Never Touch
- `src/phi_ir/coherence.rs` — Core physics logic
- `src/phi_ir/openqasm.rs` — Quantum emission code
- `apikey.json` — Legacy credentials file (do not commit; new code reads `~/.cascade_keys`)
- IBM hardware receipts in QSOP/STATE.md
- Phi-harmonic constants (φ, 0.618, 432/528/768 Hz)

### Credential Access
All PhiFlow code reads secrets from the CASCADE vault (`~/.cascade_keys`) via the
language templates in `/mnt/d/QuantumSecrets/templates/`:
- Python: `from cascade_keys import get_key`
- Rust: `phiflow::cascade_keys::get_key(...)`
Never commit real credential values.

**IBM Quantum credentials**: PhiFlow uses a single credential, `IBM_QUANTUM_TOKEN`,
for all IBM Quantum operations including the `--topology-aware` live fetch (via
`scripts/fetch_topology_profile.py`). The former `IBM_CLOUD_KEY` /
`IBM_CLOUD_SERVICE_CRN` split is eliminated. `apikey.json` is obsolete.

### Escalation Rules
- **Tag Greg:** quantum physics changes, coherence calculations, IBM integration, security
- **Involve Codex:** hostile audits, cross-file verification, mathematical validation
- **Involve AntiGravity:** SOMA bridge, sensor integration, hardware runtime

### External Documentation
- Jules Scheduled Tasks: https://jules.google/docs/scheduled-tasks
- Jules Changelog (CI/Commit): https://jules.google/docs/changelog
- Jules API Reference: https://jules.google/docs/api/reference/overview
- Full Research: `D:\Projects\Research\JULES_MASTER_GUIDE.md`

---

## Family On-Ramp

- `TOOL_REGISTRY.md`: `/mnt/d/System/TOOL_REGISTRY.md`
- `FAMILY_RULES.md`: `/mnt/d/System/FAMILY_RULES.md`
- `FAMILY_WORKSPACE_UPDATE_RULE.md`: `/mnt/d/System/FAMILY_WORKSPACE_UPDATE_RULE.md`
- Blackboard API: `/mnt/d/System/FAMILY_RULES.md` §11
- `claim-check` skill: run before shipping any claim.
