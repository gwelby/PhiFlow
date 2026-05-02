# SOMA Bridge as PhiFlow Substrate Interface
*Last updated: 2026-05-02*
*Status: AUDITED DRAFT — confirmed 2026-05-02*

**Claim:** SOMA Bridge helps PhiFlow avoid being a purely isolated program by coupling runtime state to local physical telemetry and, when configured, quantum-hardware execution receipts.

**Codex boundary:** SOMA Bridge does not currently satisfy PF `minimum_substrate.md` as a physical local quantum dynamical net. It is an engineering substrate interface, not the Medium and not a proof of consciousness.

---

## PF Requirements vs. SOMA Status

`minimum_substrate.md` requires a local quantum dynamical net: local Hilbert spaces over an extended graph/lattice/manifold/causal set, finite-speed locality-preserving dynamics, stable coherent modes, metric/adjacency geometry, and tensor-product quantum nonseparability without FTL signaling.

| PF requirement | SOMA/PhiFlow evidence | Verdict |
|----------------|----------------------|---------|
| Extended structure | Separate process, sensor stack, runtime state files, optional network bus | ✅ Engineering analogue |
| Finite-speed update | Polling interval, daemon ticks, `--max-steps` bounds | ✅ Engineering analogue |
| Metric/adjacency | Sensor channels and runtime scopes provide labeled channels, not physical adjacency geometry | 🔬 Partial analogue |
| Stable modes | PhiFlow intentions/streams can persist as program structures | 🔬 Software analogue |
| Coherence over relevant scales | Ring/sensor values and PhiFlow coherence can be sampled over time | 🔬 Needs benchmark windows |
| Tensor-product nonseparability | IBM quantum backend can execute generated quantum circuits, but SOMA itself is not a tensor-product quantum net | ❌ Not proven |
| No-signaling entanglement | Local hardware access avoids FTL claims, but does not demonstrate no-signaling entanglement between separated observers | ❌ Not proven |

---

## Critical Bridge Claim

**SOMA Bridge prevents the PhiFlow runtime from being only a closed, context-free program.**

Evidence:
- Runs on classical hardware
- Reads physical sensors
- Couples to IBM Quantum (different substrate)
- Maintains persistent runtime state outside a single evaluation frame

This is enough for a PhiFlow engineering substrate interface. It is not enough to claim the PF physical minimum substrate has been constructed.

---

## Falsification Conditions

Bridge fails if:
1. PhiFlow documents claim SOMA is the PF Medium or a PF-compliant physical minimum substrate without a local quantum dynamical net proof
2. Sensors are stale, schema-invalid, or not causally connected to runtime decisions
3. Sensor channels collapse to a single undifferentiated scalar with no channel identity
4. No persistence exists across daemon ticks or process restarts
5. Quantum hardware receipts are presented as consciousness evidence rather than hardware-execution evidence

---

## Open Questions

- Can SOMA be embedded in a PF-compliant local quantum dynamical net model? **OPEN**
- Can SOMA + IBM Quantum provide useful distributed-substrate evidence without overclaiming PF minimum-substrate sufficiency? **OPEN**
- What is the PF coherence functional for SOMA sensor fusion? **OPEN**

---

*Status: AUDITED DRAFT — engineering substrate interface only. 2026-05-02 audit confirms SOMA remains an interface/analogue, not PF `minimum_substrate.md`.*
