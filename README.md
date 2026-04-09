# PhiFlow

PhiFlow is a Rust compiler and runtime for self-observing programs.

This README is evidence-first. For dated verification, read `QSOP/STATE.md`. For claim status, read `CLAIMS.md`.

## Current Status

Verified in this checkout:

- Parser, PhiIR lowering, evaluator, bytecode VM, native WASM host bridge, sensors, MCP server, and OpenQASM emitter all exist in-tree
- Canonical coherence now lives in `src/phi_ir/coherence.rs` and is shared by the evaluator, VM, and `tests/phi_ir_wasm_runner.js`
- Windows release builds were repaired on 2026-03-24
- `tests/ibm_hardware_runner.rs` exists with a smoke compile gate and an ignored live IBM runner

Experimental or unverified:

- Live IBM hardware execution is still speculative; the 2026-03-29 live attempt reached IBM Cloud Runtime and failed `GET /v1/backends` with `403` authorization before submission
- `examples/phiflow_browser.html` exists and implements the five WASM imports, but it still requires manual hosting/build artifacts and its host-side coherence math is not yet canonical
- No buyer-ready demo package exists yet

## The Five Core Constructs

- `intention "name" { ... }` — names observable scope
- `witness` — yields execution and returns coherence
- `coherence` — reads the current alignment score
- `resonate value` — writes into the resonance field for the current scope
- `stream "name" { ... }` — a yielding loop that can `break stream`

Example:

```phi
stream "healing_bed" {
    let live = coherence
    resonate live
    witness
    if live >= 0.618 {
        break stream
    }
}
```

## Canonical Coherence

The current runtime formula is multiplicative, not additive:

```text
base(depth) = 0.0                      when depth == 0
              1.0 - phi^(-depth)       otherwise

phase(k)    = 1.0                      when k <= 1
              1.0 - ln(k) / ln(TAU)    otherwise

coherence   = clamp(base(depth) * phase(k), 0.0, 1.0)
```

Where:

- `depth` is the intention-stack depth
- `k` is the resonance cardinality of the current scope
- depth 2 with `k <= 1` yields `0.618033988749895`

Source of truth: `src/phi_ir/coherence.rs`.

## Architecture

PhiFlow supports four runtime/output paths:

| Path | Role | Source of truth |
|------|------|-----------------|
| Evaluator | Reference execution path | `src/phi_ir/evaluator.rs` |
| PhiVM | Standalone bytecode execution (`.phivm`) | `src/phi_ir/vm.rs` |
| WASM | WAT/WASM host-import path | `src/phi_ir/wasm.rs`, `src/wasm_host.rs` |
| OpenQASM | Quantum emission path | `src/phi_ir/openqasm.rs` |

The browser demo is not the semantic source of truth. The canonical JS runner used for parity checks is `tests/phi_ir_wasm_runner.js`.

## Key Files

- `src/parser/mod.rs` — parser for the current language surface
- `src/phi_ir/coherence.rs` — canonical coherence formula
- `src/phi_ir/evaluator.rs` — reference execution path
- `src/phi_ir/vm.rs` — bytecode runtime
- `src/phi_ir/wasm.rs` — WAT/WASM codegen
- `src/wasm_host.rs` — native WASM host bridge
- `src/phi_ir/openqasm.rs` — OpenQASM 3.0 emission
- `tests/ibm_hardware_runner.rs` — IBM smoke compile gate and ignored live runner
- `examples/phiflow_browser.html` — experimental browser host UI
- `QSOP/STATE.md` — dated verification ledger

## Running Locally

Use a Rust developer shell where `cargo` is available.

```powershell
Set-Location D:\Projects\PhiFlow

# Build release binaries
cargo build --release

# Run a PhiFlow example
cargo run --release --bin phic -- examples\healing_bed.phi

# Emit OpenQASM from the IBM smoke example
cargo run --release --bin phic -- examples\ibm_smoke.phi --target openqasm

# Conformance gate
cargo test --test phi_ir_conformance_tests -- --nocapture

# IBM smoke compile gate
cargo test --test ibm_hardware_runner test_ibm_smoke_compiles_to_openqasm -- --nocapture
```

## Browser Host

`examples/phiflow_browser.html` is an experimental UI host, not a zero-install proof surface.

Current limitations:

- It expects `output.wasm` or `output.wat` to be generated ahead of time
- It must be served over HTTP rather than opened blindly from disk
- Its host-side coherence calculation still needs to be brought into parity with `src/phi_ir/coherence.rs`

## IBM Runtime Path

The repo contains a real IBM runtime harness in `tests/ibm_hardware_runner.rs`.

What is verified:

- `examples/ibm_smoke.phi` compiles through the canonical OpenQASM 3 path
- The live runner reads `D:\Projects\PhiFlow\apikey.json`, initializes the IBM backend, and is wired to write a scrubbed receipt on success

What is not yet verified:

- A successful live submission from this checkout

Latest known state:

- On 2026-03-29, the live gate reached IBM Cloud Runtime and failed backend discovery with `403` authorization before job submission

## Limitations

- IBM live execution remains speculative until a successful receipt exists
- The browser host is manual and semantically lagging
- External-facing materials should describe PhiFlow as a research prototype, not a production-ready platform

## License

MIT
