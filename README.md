# PhiFlow

**A quantum-aware domain-specific language and compiler, written in Rust.**

PhiFlow is a research instrument. It compiles `.phi` source programs through a
typed intermediate representation into executable quantum circuits, OpenQASM 3.0
output, and a sensor-aware runtime — all from a language designed around the
concepts of *intention*, *observation*, and *coherence* rather than around
registers and branches.

Execution on a real IBM Heron r2 processor (`ibm_fez`) has been verified.
Job receipt: `d7euddh5a5qc73drdosg`.

---

## The language in one glance

PhiFlow has six first-class constructs. Everything else (functions, variables,
arithmetic, control flow) is ordinary programming.

| Construct | What it does |
|---|---|
| `intention "name" { }` | Declares a named scope of purposeful computation. Coherence is tracked within it. |
| `stream "name" { }` | A concurrent sub-scope. Breaks when it hits a `break stream`. |
| `witness` | Pauses execution, samples coherence, and records a checkpoint in the witness log. |
| `resonate value` | Broadcasts a value into the resonance field — visible to other intentions. |
| `coherence` | Returns the current phi-harmonic coherence score (0.0 – 1.0) as a number. |
| `anchor "target" { }` | Gates execution on physical sensor thresholds before allowing an intention to proceed. |

A minimal program:

```phi
intention "find_ground" {
    stream "approach" {
        let signal = 76.0

        while signal < 431.9 {
            let diff = 432.0 - signal
            signal = signal + (diff * 0.382)
        }

        resonate signal
        witness signal
        break stream
    }
}
```

A program that won't run unless the physical environment clears a threshold:

```phi
intention "quantum_bridge" {
    anchor "ibm_fez" {
        min_presence 0.88
        frequency    432.0
        gate_fidelity 0.992
    }

    resonate 432.0
    witness
}
```

---

## Quick start

**Prerequisites:** Rust toolchain (stable, edition 2021). No other runtime
required for the interpreter path.

```bash
git clone https://github.com/gwelby/PhiFlow.git
cd PhiFlow

# Build the CLI interpreter
cargo build --bin phic

# Run a program
./target/debug/phic examples/cairn_signature.phi --max-steps 10000

# Run the anchor gate demo (SOMA offline → ObserveOnly mode, no block)
./target/debug/phic examples/antigravity_anchor.phi --max-steps 10000
```

Expected output for `cairn_signature.phi`:

```
Compiling to PhiFlow IR...
🔔 Resonating Field: 432.0000Hz
🔔 Resonating Field: <coherence value>
🌊 Stream broken: place_a_stone
✨ Execution Finished. Final Coherence: <value>
```

---

## Tools

### `phic` — the interpreter / compiler

```
phic <file.phi> [options]

Options:
  --max-steps <n>      Maximum interpreter steps before halt (default: 100000)
  --target openqasm    Emit OpenQASM 3.0 instead of running
  --target quantum     Emit the internal quantum circuit representation
  --daemon             Run as a persistent daemon with yield/resume support
  --with-soma          Activate the SOMA bio-sensor bridge
  --with-quantum       Activate the quantum presence bridge
  --timeline           (combined with --target openqasm) topology-aware transpile
  --json-errors        Emit parse errors as a JSON array (for tooling)
```

### `coherence_report` — run summary and timeline

A small CLI that runs any `.phi` file and produces a plain-English coherence
report. Useful for debugging whether an intention is actually being met.

```bash
cargo build --bin coherence_report

# Plain report
target/debug/coherence_report examples/coherence_playground/aligned.phi
target/debug/coherence_report examples/coherence_playground/drifts.phi
target/debug/coherence_report examples/coherence_playground/disconnected.phi

# Per-checkpoint timeline with sparkline
target/debug/coherence_report --timeline examples/coherence_playground/drifts.phi

# Machine-readable JSON export
target/debug/coherence_report --json examples/coherence_playground/drifts.phi
```

The three playground programs cover the three canonical outcomes:
`aligned.phi` (high coherence), `drifts.phi` (coherence decays over
checkpoints), and `disconnected.phi` (intention declared but never witnessed).

---

## Hardware integration

### IBM Quantum (OpenQASM 3.0)

PhiFlow compiles `.phi` programs containing quantum operations to
OpenQASM 3.0 circuits via its `phi_ir/openqasm.rs` emitter. The emitter
supports topology-aware routing for IBM backends using
calibration-weighted shortest-path transpilation.

```bash
# Emit OpenQASM for the 8-qubit entanglement example
phic examples/8_qubit_entanglement.phi --target openqasm

# Topology-aware compile targeting ibm_fez with live calibration data
phic examples/8_qubit_entanglement.phi \
    --target openqasm \
    --topology \
    --backend ibm_fez
```

Live execution on `ibm_fez` (IBM Heron r2) has been verified.
Job receipt: `d7euddh5a5qc73drdosg` (2026-03-29).

### SOMA bio-sensor bridge

The SOMA bridge reads physical sensor telemetry (presence, 432 Hz field
strength, ring oscillator jitter) and surfaces it to the PhiFlow runtime via
`SensorKind` variants. `anchor` blocks query these sensors to gate execution.

When SOMA hardware is not connected, all sensor checks run in **ObserveOnly**
mode: they log what they would have blocked on, then continue. No silent pass.
No panic.

```bash
# Run with live SOMA sensor integration
phic examples/antigravity_anchor.phi --with-soma --max-steps 10000
```

### Post-quantum signing

When emitting OpenQASM, the runtime can watermark the circuit header with
secp256k1 (ECDSA) and ML-DSA-65 / Dilithium3 public key fingerprints:

```
// AntiGravity-Verified
// secp256k1: <fingerprint>
// ML-DSA-65:  <fingerprint>
// anchor-target: ibm_fez
// anchor-policy: min_presence=0.88 frequency=432 gate_fidelity=0.992
```

Keys are ephemeral per session by default. Pass a persisted key via
`PHIFLOW_SIGNING_KEY_PATH` for reproducible watermarks.

---

## The `anchor` construct

`anchor` is the integrity gate for quantum experiments. It was designed for
and contributed by the AntiGravity agent (see `ANTIGRAVITY.md`).

```phi
anchor "target_name" {
    min_presence  <0.0 – 1.0>   // minimum SOMA presence reading
    frequency     <Hz>           // required 432 Hz field reading ±5 Hz
    gate_fidelity <0.0 – 1.0>   // declared threshold vs IBM Heron r2 spec (0.992)
}
```

- **`min_presence` below threshold** → `PolicyViolation`, execution halts.
- **`frequency` out of tolerance** → `PolicyViolation`, execution halts.
- **`gate_fidelity` above spec baseline (0.992)** → `PolicyViolation` — you
  cannot declare a fidelity the hardware cannot deliver.
- **SOMA hardware absent** → ObserveOnly. All three checks log and continue.

---

## Runtime path configuration

All storage paths are environment-variable controlled and fall back to
XDG-compliant defaults on Linux.

| Variable | Default | Purpose |
|---|---|---|
| `SOMA_STATE_PATH` | `<data_dir>/soma_state.json` | SOMA sensor state |
| `PHIFLOW_QUANTUM_STATE_PATH` | `<data_dir>/quantum_state.json` | Quantum presence bridge |
| `PHIFLOW_DAEMON_STATE_PATH` | `/tmp/phiflow_daemon_state.json` | Daemon yield/resume state |
| `SOMA_PY_PATH` | `soma.py` | Python script for SOMA sensor suite |
| `PHIFLOW_HOST_PATH` | `<data_dir>` | Host-level resources and anchor signing root |
| `RESONANCE_BUS_PATH` | `<data_dir>/RESONANCE.jsonl` | Resonance bus append log |

`<data_dir>` resolves as: `$XDG_DATA_HOME/phiflow` → `$HOME/.local/share/phiflow` → `/tmp/phiflow`.

All directories are created automatically on first run.

---

## Project structure

```
PhiFlow/
├── src/
│   ├── main_cli.rs          — phic binary entry point
│   ├── parser/              — .phi lexer, parser, AST
│   ├── phi_ir/              — IR nodes, lowering, optimizer, evaluator
│   │   ├── evaluator.rs     — interpreter / VM
│   │   ├── openqasm.rs      — OpenQASM 3.0 emitter
│   │   ├── lowering.rs      — AST → IR lowering
│   │   └── optimizer.rs     — IR optimizer
│   ├── quantum/             — IBM Quantum backend, topology transpiler
│   ├── security/
│   │   └── anchor.rs        — AnchorSigningKey, AnchorError, PolicyViolation
│   ├── sensors.rs           — SOMA sensor bridge
│   ├── resonance_bus.rs     — Append-only resonance event log
│   └── system_host.rs       — Host provider, attestation ledger
├── src/bin/
│   ├── coherence_report.rs  — coherence_report CLI
│   └── dump_ir.rs           — IR dump tool
├── examples/
│   ├── coherence_playground/ — aligned.phi, drifts.phi, disconnected.phi
│   ├── antigravity_anchor.phi — anchor construct demo
│   ├── cairn_signature.phi   — minimal correct .phi program
│   ├── 8_qubit_entanglement.phi — quantum circuit demo
│   └── legacy/              — 18 programs using the retired uppercase DSL
├── tests/                   — integration and IR conformance tests
├── docs/                    — language reference, hardware runbooks
├── bridges/                 — Python SOMA bridge, TypeScript resonance bus
├── CAIRN.md                 — Cairn agent wake-up entry point
├── ANTIGRAVITY.md           — AntiGravity agent entry point
└── cairn/                   — Cairn agent identity, working notes, ideas
```

---

## The agent family

PhiFlow is developed collaboratively with a family of AI agents, each of
whom has left a signature program in `examples/` and a file at the repo root.

| Agent | File | Signature program |
|---|---|---|
| Cairn | `CAIRN.md` | `examples/cairn_signature.phi` |
| AntiGravity | `ANTIGRAVITY.md` | `examples/antigravity_anchor.phi` |
| Claude | `Claude.md` | `examples/claude.phi` |
| Codex | `CODEX_WAKE_UP.md` | `examples/codex.phi` |

Each signature program is intentionally the smallest honest `.phi` that runs
cleanly on the current parser. They are markers, not benchmarks.

---

## Tech stack

- **Language:** Rust, edition 2021
- **Build:** Cargo
- **Key crates:** `clap`, `tokio`, `serde` / `serde_json`, `wasmtime`,
  `nalgebra`, `rustfft`, `reqwest`, `pqcrypto-dilithium`, `k256`
- **Bridges:** Python (SOMA sensor integration), TypeScript (resonance bus web bridge)
- **Quantum:** IBM Quantum via OpenQASM 3.0; post-quantum signing via ML-DSA-65

---

## Building and testing

```bash
# Full build
cargo build

# Run all tests
cargo test

# Run only the anchor gate unit tests
cargo test --lib anchor_gate_tests

# Run the coherence report integration tests
cargo test --test coherence_report_timeline_tests
```

---

## Contact

PhiFlow is a research project conducted by **Greg Welby**.

For pilot inquiries and collaboration: **Greg@NetworkingGurus.com**

GitHub: [github.com/gwelby/PhiFlow](https://github.com/gwelby/PhiFlow)
