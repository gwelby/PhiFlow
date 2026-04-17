# AGENTS.md: PhiFlow
*[Workspace Type: Product | Platform | Research | Consciousness]*
*Last updated: 2026-04-16 (Truth-Sync: SOMA Bridge + Hardware Verified)*

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

## Current State
| Component | Status | Notes |
|-----------|--------|-------|
| Parser | ✅ | Handles five core constructs + block comments + imports |
| PhiIR + Lowering | ✅ | SSA IR with consciousness nodes |
| Evaluator / VM | ✅ | Backends synchronized; shared coherence math |
| WASM Codegen | ✅ | Native WASM path covered by conformance tests |
| OpenQASM 3.0 | ✅ | Native Heron-ISA [rz, sx] transposition verified |
| SOMA Bridge | ✅ | Live telemetry from `soma_state.json` verified |
| IBM Live Run | ✅ | Job `d7euddh5a5qc73drdosg` completed on `ibm_fez` 2026-04-14 |
| Browser Host | ⚠️ | Experimental; requires manual artifact hosting |
| Singularity Daemon | ✅ | T-009/T-010 complete. Daemon self-hosts with SOMA and MQTT bus |

## Income State
- Income tier: 1-3 months (Pilot-Ready)
- Single blocker: Finalizing "Buyer-Safe Pilot Offer" (T-005)
- See: `BUSINESS.md` for full income state

## What's Being Tested
- C-10: Quantum hardware execution → ✅ CONFIRMED (2026-04-14)
- C-16: Agentic reasoning as stream → 🔬 SPECULATIVE
- See: `CLAIMS.md` for full claim registry

## Identity State
- Self-reference loop: ✅ Gate 3 OPEN (Persistent Daemon with SOMA Bridge)
- Ceremony log: Initialized 2026-04-16
- See: `SOUL.md` for full identity state

## Open Questions
- Can the PhiVM daemon sustain indefinite coherence without manual reset?
- What is the specific error-cost functional for SOMA-driven quantum circuits?

## Key Commands
```powershell
# Run canonical verification gate (all truth tests)
./scripts/verify_truth.ps1

# Build release binaries
cargo build --release

# Run a PhiFlow example (SOMA bridge)
cargo run --release --bin phic -- examples/p1_soma_bridge.phi
```

## Agent Roster
| Agent | Does | Owns |
|-------|------|------|
| Greg Welby | Conductor | Architecture, Integration |
| Claude / Codex | Hardener | Parser, Compiler, VM, Tests |
| AntiGravity | Pipe-Builder | IBM Runtime, SOMA Bridge, Physics |
| Lumi (Gemini) | Protocol-Weaver | QSOP, Standards, Cleanup |
| Jules | CI/CD | Async fixes, GitHub tasks |

## Non-Negotiable Rules
1. **Read `QSOP/STATE.md` before touching code.**
2. **0.618 is derived.** Multiplicative coherence is the repo truth.
3. **No receipt = speculative.** IBM runs must be verified with job IDs.
4. **Three-backend equivalence is sacred.** Evaluator == VM == WASM.
