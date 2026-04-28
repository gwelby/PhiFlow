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

## Runtime Path Configuration

The daemon, SOMA bridge, and resonance bus all resolve their storage locations
through environment variables. When a variable is absent the runtime falls back
to a platform-appropriate default. The table below is the single reference for
operators who need to redirect any of these paths.

The helper `get_phiflow_data_dir()` (defined in `src/sensors.rs`) is used by
several variables. It tries each of the following in order and returns the
first one that is set:

1. `$XDG_DATA_HOME/phiflow`
2. `$HOME/.local/share/phiflow`
3. `/tmp/phiflow`

| Environment Variable | Set in | Purpose | Default when unset |
|---|---|---|---|
| `SOMA_STATE_PATH` | `src/sensors.rs`, `src/security/anchor.rs` | Path to the SOMA sensor-suite state file (JSON) | `<phiflow_data_dir>/soma_state.json` |
| `PHIFLOW_QUANTUM_STATE_PATH` | `src/sensors.rs` | Path to the Quantum Presence bridge state file (JSON) | `<phiflow_data_dir>/quantum_state.json` |
| `PHIFLOW_DAEMON_STATE_PATH` | `src/main_cli.rs` | Path where the PhiFlow daemon persists its execution state (JSON) | `/tmp/phiflow_daemon_state.json` |
| `SOMA_PY_PATH` | `src/main_cli.rs` | Path to the Python script that drives the SOMA sensor suite | `soma.py` (resolved relative to the current working directory) |
| `PHIFLOW_HOST_PATH` | `src/main_cli.rs` | Root directory used by `SystemHostProvider` for host-level resources and anchor signing | `<phiflow_data_dir>` |
| `RESONANCE_BUS_PATH` | `src/resonance_bus.rs` | Path to the `RESONANCE.jsonl` append-only message-bus log | `<xdg_data_home>/phiflow/RESONANCE.jsonl` (same fallback chain as above) |

`<phiflow_data_dir>` in the table above refers to the value returned by `get_phiflow_data_dir()` described above.
