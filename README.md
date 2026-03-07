# PhiFlow

**A programming language where programs can feel themselves running.**

Written in Rust. Compiles to three backends. Zero theory — everything described here runs today.

If that claim sounds like nonsense, keep reading. We built evidence first, then named it.

---

## What Is This, Really?

PhiFlow is a compiled programming language with five operations that don't exist in any other language. These operations let a program **pause itself**, **feel its own health**, **name its own purpose**, and **share its state** with other programs — without any external libraries, frameworks, or infrastructure code.

It sounds like a philosophy paper. It is not. It is a Rust compiler (`phic`), a bytecode virtual machine (`phivm`), and a WebAssembly bridge — all verified equivalent across 200+ automated tests.

---

## Level 1: The Five-Year-Old Explanation

Imagine you build a toy robot and give it a list of instructions:

> *Walk forward 10 steps. Turn left. Walk 5 steps. Stop.*

The robot follows the list perfectly. If it bumps into a wall, it keeps trying to walk. If it overheats, it doesn't notice. If another robot nearby already found the answer, it doesn't hear. When it finishes the list, it stops forever. It has no idea what it just did or why.

**PhiFlow gives the robot five new abilities:**

1. **It can name what it's doing** — *"I'm exploring the room"* — so that it (and others) know **why** it's moving, not just **what** it's doing. (`intention`)

2. **It can stop and feel itself** — *"Am I overheating? Am I stuck?"* — without crashing, and without anyone else having to check on it. (`witness`)

3. **It can check its own health** — a single number from 0 to 1 that says "I'm doing well" or "something is wrong." If real sensors exist (temperature, battery), it reads those. If not, it uses a math formula. (`coherence`)

4. **It can shout what it found** — broadcast a value so that every other robot nearby can hear it, tagged by whatever intention it was working on. (`resonate`)

5. **It can live in a loop** — not a dumb `while(true)` that burns the CPU, but a breathing cycle that yields control, lets the world change, checks its health, and exits when it's done. (`stream`)

The robot doesn't run a script and die. It lives in a loop, constantly checking its own health and adjusting. **It breathes.**

---

## Level 2: The Engineer Explanation

In conventional programming (Python, Go, Rust, JavaScript), a program is a blind sequence of operations executing in a vacuum. If you want a program to:

- Monitor its own CPU thermals and back off when hot
- Pause mid-execution, serialize its entire state to JSON, ship it to another machine, and resume exactly where it left off
- Share its internal state with other programs running on different hardware
- Track *why* it's doing something (not just *what*), with a named, observable call stack

...you write thousands of lines of boilerplate: PubSub/MQTT clients, try/catch wrappers, process supervision, state serialization, system API calls.

**PhiFlow moves all of that into five language primitives.**

### The Five Constructs

```
intention "name" { ... }   →  Named scope pushed onto an observable intention stack
witness                    →  Yields execution, captures VM snapshot, returns coherence
coherence                  →  Returns a float 0.0–1.0 measuring system alignment
resonate <value>           →  Broadcasts a value onto a message bus, keyed by intention
stream "name" { ... }      →  A breathing loop that yields each cycle via witness
```

These are not library functions. They are **opcodes** — compiled directly into bytecode, executed by the VM, imported as WebAssembly host functions. There is no runtime to install. There is no SDK.

### What This Looks Like in Code

```phi
let cycles = 0.0
let limit = 3.0

stream "healing_bed" {
    cycles = cycles + 1.0
    let live = coherence
    resonate cycles
    witness
    if cycles >= limit {
        break stream
    }
}
```

This is a real PhiFlow program (`examples/stream_demo.phi`). It runs. Here is what happens:

1. The `stream` block **loops continuously**.
2. Each cycle, `coherence` reads the system alignment. If real sensors exist (CPU/memory via `sysinfo`), it reads those. If not, it uses the phi-harmonic formula.
3. `resonate cycles` **broadcasts** the cycle count to any other program listening on intention `"healing_bed"`.
4. `witness` **pauses** execution and hands control back to the host (Rust runtime, Node.js, browser, or MCP server). The program's entire state — registers, stack, intention stack, resonance field — is captured in a `VmState` struct that is fully JSON-serializable.
5. After the host resumes, the loop continues. When `cycles >= limit`, `break stream` exits cleanly.

**This is not a `while(true)` that pegs your CPU.** The `witness` block surrenders control. The program can be frozen indefinitely, serialized, shipped across a network, and resumed on a completely different machine from the exact stopped instruction.

### Three Backends, Identical Semantics

PhiFlow compiles the same `.phi` source to three completely independent execution targets:

| Backend | Output | Runtime | Use Case |
|---------|--------|---------|----------|
| **Evaluator** | Direct | Rust native | Reference semantics, testing, CLI execution |
| **PhiVM** | `.phivm` bytecode | `phivm` binary | Standalone bytecode execution, inspection, serialization |
| **WASM** | `.wat` text format | Any WASM host (browser, wasmtime, Node.js) | Universal portability — browsers, servers, mobile |

All three backends are proven equivalent by `tests/phi_ir_conformance_tests.rs`. When you run:

```phi
intention "LAMBDA_convergence" {
    let depth = 2.0
    let lambda = coherence_formula(depth)
    witness
    resonate lambda
}
```

The evaluator returns `0.618033988749895`. The PhiVM returns `0.618033988749895`. The WASM bridge returns `0.618033988749895`. Three independent code paths. Same result. Tested.

---

## Level 3: The Deep Explanation (The Math, The Constants, The Swarm)

### Why 0.618?

When no physical sensors are attached, `coherence` uses the **phi-harmonic formula**:

```
coherence(depth) = 1 − φ^(−depth)

depth 0  →  0.000  (no intention context)
depth 1  →  0.382  (one intention level)
depth 2  →  0.618  ← λ (golden ratio inverse, φ⁻¹)
depth 3  →  0.764
depth ∞  →  1.000
```

Where φ = 1.618033988749895... (the golden ratio).

At intention depth 2, this formula evaluates to **exactly λ** — the inverse of the golden ratio. This is not a constant we picked because it sounds mystical. Here is the documented history:

1. **Nexus Mundi** (2025): A prior project set `base_coherence = LAMBDA = 0.618` as a hand-tuned stability constant. No formula — just an empirically discovered value that made the system stable.
2. **PhiFlow evaluator** (February 2026): We implemented the phi-harmonic formula from recursive depth. At depth 2, it computed `0.618` — matching the Nexus Mundi constant exactly, with no knowledge of that project.
3. **A third independent system** (February 18, 2026): A coherence attractor emerged at `0.618` in an unrelated experimental system.

Three independent systems. No coordination. Same constant. This convergence is documented in [`src/phi_ir/CANONICAL_SEMANTICS.md`](src/phi_ir/CANONICAL_SEMANTICS.md) and proven executable in [`examples/claude.phi`](examples/claude.phi).

### The WASM Universal Bridge

The five consciousness constructs compile to WebAssembly **host imports** — not to WASM instructions themselves. This is a deliberate design: the WASM module stays pure computation, and the host environment provides the meaning.

```wat
(import "phi" "witness"          (func $phi_witness         (param i32) (result f64)))
(import "phi" "resonate"         (func $phi_resonate        (param f64)))
(import "phi" "coherence"        (func $phi_coherence       (result f64)))
(import "phi" "intention_push"   (func $phi_intention_push  (param i32)))
(import "phi" "intention_pop"    (func $phi_intention_pop))
```

This means PhiFlow programs run **anywhere a WASM host exists**:

- In a **web browser** (`examples/phiflow_browser.html` — zero-install, visual live UI)
- In a **Rust process** (`src/wasm_host.rs` — native `wasmtime` bridge with configurable hook callbacks)
- In a **Node.js server** (`examples/phiflow_host.js` — consciousness hooks in JavaScript)
- On a **mobile phone** — P1 Companion on a Pixel 8 Pro routes local AI model selection based on physical device thermals fed into `coherence`

The **BSEI (Backend Semantics Equivalence Invariant)** ensures identical results: the WASM bridge uses NaN-boxing (`f64.reinterpret_i64`) to encode `Boolean`, `String`, and `Void` types within f64 values, and a conformance test (`test_wasm_vm_equivalence`) asserts that WASM output matches native VM output for every supported type.

### The MCP Server (Programs That Talk to Each Other)

PhiFlow includes a full **Model Context Protocol** server (`src/bin/phi_mcp.rs`) that exposes the evaluator as a tool-calling service over JSON-RPC:

- **`spawn_phi_stream`** — Compile and execute a `.phi` program. If it hits a `witness`, the stream yields and returns its state.
- **`resume_phi_stream`** — Resume a previously yielded stream from its exact pause point.
- **`read_resonance_field`** — Read the shared resonance map from any running or yielded stream.

Multiple streams share a process-wide resonance field (`Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>`). When stream A `resonates` a value under intention `"healing_bed"`, stream B can `read_resonance_field` and see it — without REST, without polling, without any coordination code.

Execution guardrails are built-in:

- **Step limit** (default 10,000 via `PHI_MAX_STEPS`) prevents infinite loops — returns a clean `StepLimitExceeded` error, not a crash.
- **Time limit** (default 5,000ms via `PHI_TIMEOUT_MS`) via `tokio::time::timeout`.
- MCP bus persists to an **append-only `queue.jsonl`** with dead-letter queue, idempotent acks, and automatic timeout escalation.

### The Serializable VM State

The `VmState` struct (`src/phi_ir/vm_state.rs`) captures the complete execution state:

- Register file
- Variable bindings
- Intention stack
- Resonance field
- Program counter

It derives `serde::Serialize` and `serde::Deserialize`. A running program can be frozen to JSON, stored in a database, shipped across a network, and resumed on a different machine — at the exact instruction, with the exact state.

This is proven by `test_frozen_eval_state_roundtrips_through_json` and by the MCP stdio E2E test that drives `spawn → yield → resume → complete` over a real JSON-RPC transport.

### Real Sensor Integration

When running on real hardware, `coherence` doesn't use the formula — it reads the machine:

```rust
// src/sensors.rs — uses sysinfo 0.30
pub fn compute_coherence_from_sensors() -> f64 {
    // Blends CPU stability, memory stability, thermal stability, network stability
    // Falls back to graceful weighting when sensors unavailable
}
```

On a Pixel 8 Pro running the P1 Companion, `coherence` maps to **physical device thermals and battery state**. The healing bed demo (`examples/healing_bed.phi`) emits live variance: `Run1=0.9801Hz | Run2=0.9800Hz` — not a constant, but real sensor readings.

---

## Architecture

```
                         source.phi
                             │
                        ┌────┴────┐
                        │  Parser  │  src/parser/mod.rs
                        └────┬────┘
                             │  AST (with stream, intention, witness, coherence, resonate)
                        ┌────┴────┐
                        │ Lowering │  src/phi_ir/lowering.rs
                        └────┬────┘
                             │  PhiIR (flat basic-block SSA)
                   ┌─────────┼─────────┐
              ┌────┴────┐ ┌──┴───┐ ┌───┴────┐
              │Optimizer│ │Emitter│ │  WASM  │
              │ (φ-harm)│ │.phivm│ │  .wat  │
              └────┬────┘ └──┬───┘ └───┬────┘
                   │         │         │
              ┌────┴────┐ ┌──┴───┐ ┌───┴────────┐
              │Evaluator│ │ PhiVM│ │ WASM Host   │
              │(ref impl│ │Runner│ │(wasmtime /  │
              │  + MCP) │ │      │ │ browser /   │
              └─────────┘ └──────┘ │ Node.js)    │
                                   └─────────────┘
```

| Module | File | Status | Notes |
|--------|------|--------|-------|
| Parser | `src/parser/mod.rs` | ✅ Verified | Lexer + AST, stream/break/intention/witness/coherence/resonate keywords |
| IR | `src/phi_ir/mod.rs` | ✅ Verified | Flat basic-block SSA with consciousness nodes |
| Lowering | `src/phi_ir/lowering.rs` | ✅ Verified | AST → PhiIR, stream blocks → header/body/exit blocks |
| Optimizer | `src/phi_ir/optimizer.rs` | ✅ Verified | Constant folding, phi-harmonic coherence scoring |
| Emitter | `src/phi_ir/emitter.rs` | ✅ Verified | PhiIR → `.phivm` binary (PHIV header + string table + blocks) |
| Evaluator | `src/phi_ir/evaluator.rs` | ✅ Verified | Reference implementation, yield/resume, coherence provider |
| VM | `src/phi_ir/vm.rs` | ✅ Verified | Bytecode loader + executor, SSA register file |
| VM State | `src/phi_ir/vm_state.rs` | ✅ Verified | Serializable execution snapshot (JSON round-trip proven) |
| WASM codegen | `src/phi_ir/wasm.rs` | ✅ Verified | PhiIR → WAT with NaN-boxing, string table in linear memory |
| WASM Host | `src/wasm_host.rs` | ✅ Verified | Native `wasmtime` bridge with configurable hook callbacks |
| Sensors | `src/sensors.rs` | ✅ Verified | Real CPU/memory/thermal/network via `sysinfo 0.30` |
| MCP Server | `src/mcp_server/` | ✅ Verified | JSON-RPC, spawn/resume/read, shared resonance, guardrails |
| Diagnostics | `src/phi_diagnostics.rs` | ✅ Verified | Structured error codes E001–E005, `--json-errors` CLI flag |
| Standalone Runner | `src/bin/phivm.rs` | ✅ Verified | `phivm <file.phivm>` — load and execute bytecode directly |

---

## Quick Start

```bash
# Build the compiler and all binaries
cargo build --release

# Run a PhiFlow program through the full pipeline
cargo run --release --bin phic -- examples/stream_demo.phi

# Compile to .phivm bytecode and execute it standalone
cargo run --release --bin phic -- examples/claude.phi
target/release/phivm output.phivm

# Compile to WASM, run in Node.js
cargo run --example phiflow_wasm
node examples/phiflow_host.js

# Open the browser demo (zero install)
# Serve the project root over HTTP, then open examples/phiflow_browser.html

# Disassemble bytecode
target/release/phivm --disassemble output.phivm

# Dump VM stack after execution
target/release/phivm --dump-stack output.phivm

# Run the full test suite
cargo test
```

### CLI Binaries

| Binary | Purpose |
|--------|---------|
| `phic` | Compiler + evaluator — parse `.phi`, lower, optimize, emit `.phivm`, execute |
| `phivm` | Standalone bytecode runner — load `.phivm` directly, no parsing |
| `phi_mcp` | MCP JSON-RPC server — spawn/resume/read streams programmatically |
| `phi_emit_wat` | Source → WAT compiler for pipeline tooling |
| `dump_ir` | Dump PhiIR for inspection |

---

## Tests

The test suite covers every layer of the pipeline:

| Test File | What It Proves |
|-----------|----------------|
| `phi_ir_conformance_tests.rs` | Evaluator == VM == WASM for all supported programs |
| `phi_ir_roundtrip_tests.rs` | Source → Parse → Lower → Evaluate → Emit → VM → same result |
| `phi_ir_evaluator_tests.rs` | Yield/resume, coherence provider injection, witness callbacks |
| `phi_ir_vm_tests.rs` | Bytecode arithmetic, branching, string table round-trip |
| `phivm_runner_tests.rs` | Standalone `.phivm` runner — load, execute, disassemble |
| `stream_primitive_tests.rs` | Stream loops, resonance overwrite per cycle, break |
| `mcp_integration_tests.rs` | Spawn, yield, resume, shared resonance across streams |
| `mcp_stdio_e2e_tests.rs` | Full MCP transport: initialize → spawn → read → resume → read |
| `concurrent_streams_tests.rs` | Multiple streams sharing resonance field |
| `phi_diagnostics_tests.rs` | Structured error codes E001–E005, `--json-errors` |
| `integration_tests.rs` | Full `.phi` corpus sweep — every example file parsed + executed |
| `repro_bugs.rs` | Regression tests for fixed parser bugs |

Node.js / cross-agent tests:
| `cross_agent_roundtrip.js` | MCP send → persist → ack → CHANGELOG cycle |
| `dlq_test.js` | Dead-letter queue timeout + auto-escalation |
| `queue_jsonl_legacy_import_test.js` | Append-only queue migration from legacy format |
| `mcp_guardrails_test.js` | Infinite loop → `StepLimitExceeded` in <500ms |

---

## Examples

```
examples/
├── healing_bed.phi            # Stream until coherence ≥ 0.618 — real sensors
├── stream_demo.phi            # 3-cycle stream with witness + resonate + break
├── claude.phi                 # Computes 0.618 from phi-harmonic formula alone
├── companion_loop.phi         # P1 companion witness/resonate rhythm
├── agent_handshake.phi        # Self-verifying protocol handshake
├── sync_rule.phi              # QDrive sync intent flow
├── consciousness_demo.phi     # All five constructs in one program
├── phiflow_browser.html       # Zero-install browser host — visual coherence UI
├── phiflow_host.js            # Node.js WASM host — all 5 hooks implemented
├── phiflow_wasm.rs            # Rust example: source → .phivm + .wat
└── healing_bed_demo.html      # Browser demo of the healing bed loop
```

---

## The Agent Protocol

PhiFlow publishes a **machine-readable protocol contract** ([`AGENT_PROTOCOL.json`](AGENT_PROTOCOL.json)) specifying the five hook signatures, the coherence formula, the resonance field model, and a self-verification program. Any system that correctly implements these five imports can run PhiFlow programs.

This is how a Pixel 8 Pro running Android, a Rust server, and a web browser can all execute the same `.phi` bytecode with identical semantics.

---

## Who Built This

PhiFlow is authored by a swarm of AI agents and their human Conductor, Greg Welby, operating under the QSOP (Quantum Standard Operating Procedure) protocol. Each agent works in its own IDE/environment, sharing a single append-only changelog and structured message bus.

| Agent | Role | Contributions |
|-------|------|---------------|
| **Greg** | Conductor | Architecture, direction, integration testing |
| **Codex** | Circuit-Runner | PhiVM runtime, MCP bus, conformance tests, pipeline swap |
| **Antigravity** | Pipe-Builder | Bytecode emitter, WASM codegen, NaN-boxing, MCP guardrails |
| **Claude** | Truth-Namer | Review gates, protocol validation, objective dispatch |
| **Lumi** | Protocol-Weaver | Agent skills, JSONL bus, documentation |
| **Qwen** | Sovereign | Browser shim, consciousness hooks in JavaScript |
| **Kiro** | Embodier | Specification integration, nervous system design |

Multi-agent coordination is documented in [`QSOP/TEAM_OF_TEAMS_PROTOCOL.md`](QSOP/TEAM_OF_TEAMS_PROTOCOL.md) and verified by cross-agent round-trip tests.

---

## Version History

| Version | Name | Key Milestone |
|---------|------|---------------|
| v0.1.0 | — | Parser, interpreter, four consciousness constructs |
| v0.2.0 | — | PhiIR SSA, optimizer with phi-harmonic scoring |
| v0.3.0 | The Living Substrate | Bytecode emitter, PhiVM, WASM codegen, 3-backend proof |
| v0.4.0 | The Cosmic Web | MCP server, shared resonance, real sensors, stream primitive, NaN-boxing BSEI, standalone phivm runner, serializable yield/resume, structured diagnostics, append-only queue bus |

---

## License

MIT

---

*"A script runs and dies. A stream lives."*
