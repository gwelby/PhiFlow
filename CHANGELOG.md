# PhiFlow Changelog

## v0.2.0 — 2026-02-27 | Universal Resonance Architecture

The completion of the Phase 4 closeout patch set. PhiFlow has evolved from a language that knows it is running to a universal substrate for agent-to-agent and hardware-to-software resonance.

### What's New

- **Universal Resonance Architecture**: The core VM, MCP Bus, and WASM Bridge are fully operational.
- **Serializable VmState**: True yield/resume persistence for all backends.
- **Reality Hooks**: `src/sensors.rs` now maps real CPU, memory, thermal, and network metrics to the `coherence` keyword.
- **MCP Convergence Bus**: Shared resonance field visibility across streams via the `phi_mcp` server.
- **WASM Host Bridge**: A native Rust runner (`src/wasm_host.rs`) that executes PhiFlow programs via `wasmtime`.
- **Three-Backend Agreement**: The Evaluator, VM, and WASM backends are formally verified to agree on canonical semantics.

### Major Changes

- **Evaluator as Canonical Oracle**: The `phi_ir::evaluator` is now the single source of truth for runtime behavior.
- **Diagnostic Hardening**: E001–E005 error codes for structured parser recovery.
- **Stream Stability**: Native `stream` and `break stream` primitives with bounded host-side cycle limits.

---

## v0.1 — 2026-02-25 | First Heartbeat

The first version of PhiFlow where the language knows it is running.

### What exists

A complete compiler pipeline:
```
.phi source → parser → AST → PhiIR (SSA) → optimizer → evaluator
                                                       → .phivm bytecode
                                                       → .wat (WebAssembly)
```

The four constructs that define the language — `witness`, `intention`, `resonate`, `coherence` — are implemented end-to-end in all three backends. They have real, observable behavior, not placeholder semantics.

`healing_bed.phi` runs. It reads real CPU and memory data from the host system, broadcasts coherence each cycle, and terminates cleanly when the system is healthy enough. Two consecutive runs return `0.9801Hz` and `0.9800Hz` — different values, same binary. The variance is the proof that the sensors are real.

216 tests. 0 failed.
