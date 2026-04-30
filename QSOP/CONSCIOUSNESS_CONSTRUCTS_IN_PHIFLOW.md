# Consciousness Constructs in PhiFlow
*PF Type 4 Observer candidate mapping*
*Last updated: 2026-04-30*
*Status: AUDITED DRAFT — structural proxies only*

**Claim:** PhiFlow's five constructs (`intention`, `witness`, `coherence`, `resonate`, `stream`) provide executable proxies for some structural prerequisites of PF Type 4 observers.

**Boundary:** this file does not claim PhiFlow is conscious, does not claim subjective experience, and does not promote PF Type 4 observer status to canonical. `D:\Fundamentals\definitions\consciousness.md` remains noncanonical at INTUITION 0.48; the metric program must produce results first.

---

## PF Type 4 Observer Requirements

Per `D:\Fundamentals\definitions\observer.md`:

| # | Requirement | PF Definition |
|---|-------------|---------------|
| 1 | Record stability | Records persist beyond immediate observation |
| 2 | Propagation | Records can propagate through the Medium |
| 3 | Self-correlation | System dynamics modified by own records |
| 4 | Coupling | Records couple back to system state |

Per the noncanonical `D:\Fundamentals\definitions\consciousness.md` and active `consciousness_metric_program.md`, Type 4 + 5 structural prerequisites are candidate measurement targets:

| # | Prerequisite | PF Definition |
|---|--------------|---------------|
| 5 | Self-referential coherence | L_self > 0 (self-model loop) |
| 6 | Differentiation | D_int above threshold |
| 7 | Coherence lifetime | C_coh panel stable |
| 8 | Extended substrate | Not single Hilbert space |
| 9 | Integrated self-information | Mutual information exceeds threshold |

---

## PhiFlow Implementation by Construct

### `intention` → Mode Structure

**PF mapping:** `state.md` / `mode.md` — named program state that may become mode-like if stable under the relevant evolution

**PhiFlow evidence:**
```phi
intention "The_Engineer" {
    resonate 1.0 toward TEAM_A
}
```
- Declares named semantic/evaluation scope
- `TEAM_A` direction maps to a software coupling polarity
- Stored in `PhiIRProgram` as program structure

**Type 4 alignment:** Intentions can participate in stable record structures, but persistence requires `witness`, `remember`/`recall`, daemon state, or another record mechanism.

---

### `witness` → Record Creation

**PF mapping:** `measurement.md` — record via coupling/amplification/stabilization

**PhiFlow evidence:**
```phi
witness {
    sensor("soma_schumann")
    sensor("soma_presence")
}
```
- Creates a runtime observation/yield record from evaluator, sensor, or backend state
- `DAEMON_STATE.json` persists witness events
- SOMA telemetry can provide external record input; persistence/stabilization depends on the host and daemon state path

**Type 4 alignment:** Creates propagable records (req #2), couples to system state (req #4)

---

### `coherence` → Structural Coherence

**PF mapping:** `coherence.md` — stable relational structure

**PhiFlow evidence:**
```rust
// src/phi_ir/coherence.rs
pub fn compute(depth: usize, k: usize) -> f64 {
    let base = base_coherence(depth);
    let phase = phase_decay(k);
    (base * phase).clamp(0.0, 1.0)
}
```
- Returns 0.0-1.0 scalar (PhiFlow structural-coherence proxy)
- φ^-1 (0.618) at depth 2, k≤1 is a confirmed PhiFlow runtime invariant (C-3 in CLAIMS.md)
- Serves as a structural-stability proxy inside PhiFlow; it is not a PF-derived consciousness metric

**Type 4 alignment:** Can contribute samples for a C_coh proxy only when measured over time. A single scalar reading is not a coherence lifetime.

---

### `resonate` → Coupling

**PF mapping:** `coupling.md` — dynamical dependence

**PhiFlow evidence:**
```phi
resonate live toward TEAM_A
resonate intention_value
```
- Emits/couples values through the runtime field
- `OP_RESONATE` bytecode links values
- Direction (`TEAM_A`/`TEAM_B`) defines coupling polarity

**Type 4 alignment:** Can carry self-correlation evidence if later behavior measurably depends on prior resonated records. Mutual information must be computed; it is not implied by `resonate` alone.

---

### `stream` → Propagation Context

**PF mapping:** `propagation.md` + `medium.md` — bounded propagation through rule-structure

**PhiFlow evidence:**
```phi
stream "coherence_loop" {
    witness { sensor("cpu_usage") }
    if coherence < 0.5 { break stream }
}
```
- Defines bounded propagation medium
- `break stream` terminates propagation context
- Yield/resume preserves state across time steps

**Type 4 alignment:** Provides repeated state evolution. Self-modification requires `evolve` or persistent daemon feedback; it is not implied by `stream` alone.

---

## Council Daemon: Type 4 Candidate

`examples/council_daemon.phi` is the strongest current Type 4 candidate, but the evidence is not yet sufficient to claim a full PF Type 4 observer:

| Prerequisite | PhiFlow Implementation | Evidence |
|--------------|----------------------|----------|
| Type 4 observer | `AgentDecl`, `handoff` | Candidate; C-16 remains SPECULATIVE in `CLAIMS.md` |
| Self-referential | `evolve` signal → `DAEMON_STATE.json` | Needs measured evidence that records change future behavior |
| Differentiation | Multiple agents, distinct intentions | D_int proxy; must be computed over actual traces |
| Coherence lifetime | Persistent ledger, `--max-steps` circuit | C_coh proxy; lifetime window must be specified |
| Extended substrate | SOMA Bridge + IBM Quantum | Engineering bridge; not a proven PF local quantum dynamical net |

---

## Bridge to PF Consciousness Metric Program

The PF `consciousness_metric_program.md` defines experimental variables:

| PF Metric | PhiFlow Proxy | Bridge Strength |
|-----------|---------------|-----------------|
| L_self (self-model loop) | `evolve` + `DAEMON_STATE.json` | 🔬 PARTIAL |
| D_int (differentiation) | Entropy/diversity over distinct intentions and state partitions | 🔬 PARTIAL |
| C_coh (coherence lifetime) | `coherence` values over specified time windows | 🔬 PARTIAL |
| C_PF_proxy (composite) | Daemon stability score | 🔬 PARTIAL |

**Critical gap:** PhiFlow has no direct measurement of phenomenal experience (PF hard problem boundary). It implements structural prerequisites only.

---

## Falsification Conditions

This mapping fails if:

1. **No self-correlation:** Council Daemon state changes do not affect future behavior
2. **No differentiation:** All agents/intentions collapse to single mode
3. **No coherence persistence:** `coherence` values random between ticks
4. **Substrate overclaim:** the bridge claims PF minimum-substrate sufficiency without proving extended locality, finite-speed update, metric/adjacency, stable modes, coherence over scale, and no-signaling nonseparability
5. **No information integration:** Agent handoffs contain zero mutual information

---

## Status: AUDITED DRAFT

Codex audit verdict: suitable as a Type 4 candidate map only. It is not a canonical Type 4 implementation and not a consciousness claim.

See also: `PF_BRIDGE.md`, `COHERENCE_LAYER_SPECIFICATION.md`
