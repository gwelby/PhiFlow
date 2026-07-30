# Topology-Aware Transpiler

## Implemented Shape
- Added backend-side quantum overlay analysis in `src/phi_ir/quantum_interaction.rs`
- Added backend topology profile types in `src/quantum/backend_topology.rs`
- Added ladder corridor placement in `src/phi_ir/topology_transpiler.rs`
- Added topology-aware compile surface in `src/lib.rs`
- Added additive emitter path in `src/phi_ir/openqasm.rs`
- Added live IBM topology extraction in `src/quantum/ibm_quantum.rs`

## Current Scope
- Supports contradiction-ladder recognition for `examples/cognitive_dissonance.phi`
- Supports topology-aware Heron-native `cz` emission on adjacent physical edges
- Preserves legacy `emit()` / `compile_to_openqasm()` behavior when topology is not enabled
- Adds CLI support via `--topology-aware --topology-backend ibm_fez`

## Known Boundaries
- The overlay is backend-only; evaluator / VM / WASM semantics are unchanged
- Phase 1 remains corridor-first and fail-fast; no general SWAP router is implemented
- `resonate ... as "channel"` is parsed and discarded because the current IR does not preserve channel labels
