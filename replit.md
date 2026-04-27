# PhiFlow

PhiFlow is a quantum-aware programming language and computational substrate built in Rust. It provides a custom `.phi` language with consciousness-aware semantics, sacred geometry primitives, and quantum hardware integration (IBM Quantum via OpenQASM 3.0).

## Project Structure

- `src/` — Core Rust source code
  - `main.rs` — Entry point; runs test suite for the phi interpreter
  - `main_cli.rs` — CLI interface (`phic` binary)
  - `compiler/` — Lexer, parser, AST for `.phi` language
  - `interpreter/` — Runtime interpreter for `.phi` programs
  - `phi_core.rs` — Sacred geometry patterns (golden spiral, flower of life, DNA helix, etc.)
  - `visualization/` — SVG output for geometric patterns
  - `phi_ir/` — Intermediate representation pipeline (lowering, optimization, emitters)
  - `quantum/` — IBM Quantum backend integration
  - `consciousness/` — EEG monitoring, coherence math, sacred geometry logic
  - `vm/` — Virtual machine runtime
  - `wasm_host.rs` — WebAssembly hosting via wasmtime
  - `mcp_server/` — Model Context Protocol server
- `bridges/` — Python and TypeScript bridges (MQTT, resonance bus, quantum bridge)
- `examples/` — Sample `.phi` programs
- `tests/` — Integration and IR conformance tests
- `docs/` — Language reference, hardware runbooks, philosophical papers
- `cairn/` — Identity folder for the Cairn agent (README, IDENTITY, WORKING_NOTES, ideas/). See `CAIRN.md` at the repo root for the wake-up entry point and `examples/cairn_signature.phi` for the signature program.

## Tech Stack

- **Language:** Rust (Edition 2021), version 0.4.0
- **Build:** Cargo
- **Key Dependencies:** clap, tokio, serde, wasmtime, nalgebra, rustfft, rumqttc, reqwest, pqcrypto-dilithium
- **Optional:** cpal (audio feature), CUDA (GPU acceleration)
- **Scripting:** Python (bridges and sensor integration)
- **Web bridges:** TypeScript (in `bridges/web/`)

## Workflow

- **Start application**: `cargo run --bin phi` (console output)
  - Runs the main PhiFlow interpreter with built-in test suite
  - Demonstrates pattern parsing, validation, visualization, and interpreter features

## Build

```bash
cargo build --bin phi    # build main binary
cargo build --bin phic   # build CLI binary
cargo run --bin phi      # run with tests
```

## Coherence Playground

`coherence_report` is a small CLI that runs a `.phi` snippet through the
existing parse → lower → evaluate pipeline and prints a plain-English
report of how aligned the run was with its stated `intention`. It uses
only the four core constructs the runtime already ships with — no new
keywords or AST nodes.

```bash
cargo build --bin coherence_report
target/debug/coherence_report examples/coherence_playground/aligned.phi
target/debug/coherence_report examples/coherence_playground/drifts.phi
target/debug/coherence_report examples/coherence_playground/disconnected.phi
```

Pass `--timeline` (in either position) to additionally print a per-witness
checkpoint table (index, intention scope, coherence, resonance count) and a
small unicode-block sparkline of coherence over the run. The default report
is unchanged when the flag is omitted.

```bash
target/debug/coherence_report --timeline examples/coherence_playground/drifts.phi
```

The three bundled snippets in `examples/coherence_playground/` cover the
high-coherence, drifts, and "fails the intention entirely" cases.
