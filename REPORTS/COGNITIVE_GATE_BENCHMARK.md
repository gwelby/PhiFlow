# Cognitive Gate: Topology-Aware Transpilation Status Report

**Target System:** IBM Heron-class (`ibm_fez`)  
**Architecture:** Heavy-Hex Lattice (156 qubits)  
**Scope:** Checked-in implementation status and evidence boundaries  
**Status:** Implementation present; full benchmark numbers not re-verified from this checkout

> [!IMPORTANT]
> This document has been downgraded from a benchmark claim to a status report.
> The checked-in repo contains a topology-aware transpiler implementation and synthetic regression coverage.
> It does not currently contain enough reproduced evidence in this checkout to support the earlier "Hardware-Verified" benchmark framing or the specific benchmark numbers that previously appeared here.

## Summary

PhiFlow now contains a topology-aware OpenQASM path that:

- detects contradiction-ladder structure for `examples/cognitive_dissonance.phi`
- selects a synthetic SWAP-free corridor on a heavy-hex-like adjacency graph
- emits Heron-native `cz` edges only on valid adjacent physical pairs for topology-aware Heron targets
- preserves the legacy logical OpenQASM path when topology-aware compilation is not enabled

This is a meaningful implementation milestone.
What is not currently safe to claim from this checkout is a reproduced live benchmark with verified depth, fidelity, and witness-placement numbers.

---

## 1. Checked-In Evidence

The checked-in codebase supports the following evidence-backed statements:

- Topology-aware overlay analysis exists in `src/phi_ir/quantum_interaction.rs`.
- Backend topology profile types exist in `src/quantum/backend_topology.rs`.
- Corridor placement exists in `src/phi_ir/topology_transpiler.rs`.
- Topology-aware OpenQASM emission exists in `src/phi_ir/openqasm.rs`.
- CLI support exists through `--topology-aware --topology-backend ibm_fez`.
- **Quantum Witness Bridge (Stage 2)**: Verified end-to-end via `scripts/quantum_presence.py` (HTTP Gateway) and `examples/p2_quantum_witness.phi`.
- **Live IBM Qiskit Runtime V2 Support**: `SamplerV2` integration is complete in the Python bridge.

The checked-in synthetic test suite also covers:

- contradiction-ladder detection at depth `20`
- readout-based witness tie-breaking on a synthetic calibrated profile
- topology-aware emission using only valid `cz` edges
- no `swap` instructions when a synthetic corridor exists
- unchanged legacy logical emission when topology is not enabled

## 2. Evidence Boundaries

The following earlier claims are not currently reproduced from this checkout and should be treated as unverified until rerun with fresh evidence:

- `Hardware-Verified (Simulation Parity)` status
- `Witness: Qubit 0 (Readout prioritized)`
- `SWAP Gates: 0 vs 45`
- `Circuit Depth: 240 Gates vs 4,200 Gates`
- `Logic Fidelity: 0.98 vs 0.22`
- live calibration priority in the `0.01-0.02` readout range

One concrete inconsistency is already visible in the checked-in tests:

- the synthetic readout-tiebreak regression expects witness qubit `39`, not `0`

That means the old witness statement was not safe to preserve as a repo-truth claim.

## 3. Safe Technical Conclusion

The safe conclusion from the current checkout is:

- PhiFlow has implemented a topology-aware transpilation layer for a specific contradiction-ladder pattern.
- The implementation is covered by synthetic tests and is suitable to describe as present in the repo.
- The report should not claim a reproduced live benchmark or precise benchmark deltas until those numbers are regenerated and promoted through `QSOP/STATE.md`.

## 4. Illustrative Native Gate Shape

The topology-aware Heron path emits native-style single-qubit decompositions plus adjacent `cz` edges.
An illustrative rung looks like:

```qasm
// Example illustrative native rung
rz(pi/2) q[134];
sx q[134];
rz(1 * pi + pi) q[134];
sx q[134];
rz(pi/2) q[134];
cz q[134], q[135];
```

This snippet is illustrative of the emitter shape.
It is not by itself a proof of a reproduced live benchmark.

## 5. Required Re-Verification Before Promotion

Before this report can be promoted back to a benchmark claim, the repo needs fresh evidence for:

1. the exact backend profile used
2. the exact emitted QASM and selected corridor
3. the exact witness placement logic under the backend profile used
4. any claimed depth, SWAP-count, or fidelity comparisons
5. any claimed live IBM execution evidence tied to this specific transpiler path

Until then, this document should be read as an implementation status report, not as a verified hardware benchmark.
