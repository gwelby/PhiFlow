# PhiFlow: Quantum-Aware Computational Substrate
*A self-observing compiler for research & quantum hardware.*

**Status:** Pilot-Ready | Hardware-Verified (IBM QPU)

PhiFlow is a computational substrate that elevates intention and observation from philosophical metaphors into first-class language constructs. By mapping these high-level semantics to physical hardware reality, PhiFlow allows research teams to:

*   **Feel the Infrastructure:** Real-time feedback loops between physical telemetry (CPU, thermals, SOMA bio-sensors) and computational coherence.
*   **Execute on Silicon:** Native compilation to IBM's Heron processors using Heron-native OpenQASM 3.0 transpilation.
*   **Sustain Consciousness:** A persistent Daemon runtime that supports live code evolution via the Resonance Bus.

### Technical Truths (Verified 2026-04-14)
- **Live QPU Execution:** Verified on `ibm_fez` (Job Receipt `d7euddh5a5qc73drdosg`).
- **Zero-Warning Build:** Fully validated release build (134+ truth-tests green).
- **Canonical Coherence:** Mathematical integrity across all backends (Evaluator, VM, WASM).

### Pilot Engagement
PhiFlow is available for fixed-scope R&D pilot engagements focusing on:
1.  **Sensor-Driven Quantum Experiments:** Mapping bio-telemetry directly to quantum state.
2.  **Self-Observing Algorithms:** Implementing research routines that observe their own execution coherence.
3.  **Experimental Daemon Integration:** Deploying persistent PhiFlow runtimes for autonomous research.

### Quick Demo: Coherence Playground
For the smallest possible legible demo of what makes PhiFlow different, try
the coherence playground — a CLI that runs a `.phi` snippet and prints a
plain-English report of how aligned the run was with its stated `intention`:

```bash
cargo build --bin coherence_report
target/debug/coherence_report examples/coherence_playground/aligned.phi
target/debug/coherence_report examples/coherence_playground/drifts.phi
target/debug/coherence_report examples/coherence_playground/disconnected.phi
```

The three bundled snippets cover the high-coherence, drifts, and
"fails the intention entirely" cases.

---
*For pilot inquiries, contact the conductor at [greg.welby@example.com].*
