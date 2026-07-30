# PhiFlow Architecture

**Status:** Living document
**Last updated:** 2026-07-21
**Purpose:** Developer-facing guide to how PhiFlow actually works. Complements `IR_DESIGN.md` (the IR spec) and `CANONICAL_SEMANTICS.md` (the coherence formula).

---

## 1. What PhiFlow Is (Programmer's Translation)

PhiFlow is a Rust compiler and runtime. You write `.phi` files. The compiler parses them, lowers them to an SSA-like IR (PhiIR), optimizes, and runs them. Same as any language. The twist is what the **primitives** are.

In a normal language, your primitives are `if`, `while`, `function`, `variable`, `return`. In PhiFlow, the four core primitives are:

| PhiFlow Primitive | What it literally does in the evaluator | Programmer analogy |
|---|---|---|
| `intention "name" { ... }` | Pushes `"name"` onto a stack. When the block exits, pops it. | Entering a named scope or a `with` block. The stack depth is the only state it changes. |
| `witness` | Captures a snapshot of the VM's internal state (intention stack, coherence score, registers, resonance count). Calls `host.on_witness(snapshot)` which can say "continue" or "yield" (pause the program). | A breakpoint + profiler checkpoint. The program can be **paused from outside** at this exact point. |
| `resonate value` | Appends `value` to `resonance_field[current_intention_name]`. A scoped dictionary write. | Writing to a named channel or topic. Other parts of the program can read the same field. |
| `coherence` | Returns `base(depth) × phase(k)` where depth = intention stack depth and k = how many values have been resonated in the current scope. Pure math. | Reading a metric. It's a formula. |

There are two additional constructs that build on these:

| Construct | What it does |
|---|---|
| `stream "name" { ... }` | A loop that pushes its name to the intention stack, runs its body repeatedly, and breaks when `break stream` is hit. Resonance inside a stream **overwrites** rather than accumulates (prevents memory leaks in long-running loops). |
| `anchor "target" { ... }` | Gates execution on physical sensor thresholds. The program blocks until the target sensor reads above `min_presence`. |

**That's the whole language model.** Everything else — variables, arithmetic, control flow, functions — is ordinary programming.

---

## 2. The Compilation Pipeline

```
.phi source file
       │
       ▼
┌──────────────────────────────────┐
│  PARSER (src/parser/mod.rs)      │
│                                  │
│  Tokenizer → recursive descent   │
│  Output: Vec<PhiExpression>      │
│  (the AST)                       │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  LOWERING (src/phi_ir/lowering.rs)│
│                                  │
│  AST → PhiIRProgram              │
│  - Basic blocks with labels      │
│  - SSA operands (u32 indices)    │
│  - PhiIRNode enum instructions   │
│  - String table for interning    │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  OPTIMIZER (src/phi_ir/optimizer.rs)│
│                                  │
│  Basic optimization passes       │
│  (constant folding, dead code)   │
└──────────────┬───────────────────┘
               │
       ┌───────┼───────────┐
       │       │           │
       ▼       ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────────┐
│ NATIVE  │ │  WASM   │ │   QUANTUM   │
│ Evaluator│ │  WAT    │ │  OpenQASM   │
│ src/phi_│ │  emit   │ │  3.0 emit   │
│ ir/     │ │ src/phi_│ │  src/phi_ir/│
│evaluator│ │ ir/wasm │ │  openqasm.rs│
│  .rs    │ │  .rs    │ │             │
└─────────┘ └─────────┘ └─────────────┘
```

### 2.1 Parser

The parser (`src/parser/mod.rs`) is a single-file recursive descent parser. It tokenizes the source and builds an AST of `PhiExpression` nodes.

Key tokens: `intention`, `stream`, `witness`, `resonate`, `coherence`, `anchor`, `break`, `listen`, `broadcast`, `remember`, `recall`, `evolve`, `entangle`, `handoff`, `sensor`, `field`.

The parser handles:
- Block comments (`/* ... */`)
- Type annotations (`let x: number = 42`)
- Module imports (`import from "file.phi"`)
- Nested intention/stream blocks
- Binary and unary expressions
- Function definitions and calls

### 2.2 Lowering

The lowering pass (`src/phi_ir/lowering.rs`) converts the AST into `PhiIRProgram`:

```rust
pub struct PhiIRProgram {
    pub blocks: Vec<PhiIRBlock>,       // Basic blocks in order
    pub entry: BlockId,                 // Entry block ID
    pub string_table: Vec<String>,      // Interned strings
    pub frequencies_declared: Vec<(SacredFrequency, f64)>,
    pub intentions_declared: Vec<String>,
}
```

Each `PhiIRBlock` contains:

```rust
pub struct PhiIRBlock {
    pub id: BlockId,
    pub label: String,
    pub instructions: Vec<PhiInstruction>,
    pub terminator: PhiIRNode,  // Branch, Jump, Return, or Fallthrough
}
```

Each `PhiInstruction` has an optional result operand (SSA-style):

```rust
pub struct PhiInstruction {
    pub result: Option<Operand>,  // u32 index into register file
    pub node: PhiIRNode,
}
```

### 2.3 The PhiIR Instruction Set

`PhiIRNode` is a Rust enum with ~30 variants. The key ones:

**Standard computation:**
- `Const(PhiIRValue)` — load a constant
- `LoadVar(String)` / `StoreVar { name, value }` — variable access
- `BinOp { op, left, right }` — arithmetic/comparison
- `Call { name, args }` — function call
- `Branch { condition, then_block, else_block }` — conditional jump
- `Jump(BlockId)` — unconditional jump
- `Return(Operand)` — return value

**PhiFlow-unique nodes:**
- `Witness { target, collapse_policy }` — self-observation
- `WitnessSensor { sensor }` — read physical sensor
- `IntentionPush { name, frequency_hint }` / `IntentionPop` — scope management
- `Resonate { value, frequency_relationship, direction }` — share state
- `StreamPush(String, Option<f64>)` / `StreamPop` — continuous loop
- `CoherenceCheck` — read coherence score
- `FieldCoherence` / `Dissonance` / `CoherenceOf(String)` — resonance field queries
- `Remember { key, value }` / `Recall(String)` — persistent storage
- `Broadcast { channel, value }` / `Listen(String)` — inter-agent messaging
- `Evolve(Operand)` — self-modification request
- `Entangle(f64)` — phase-locking
- `Handoff { target_agent, task_id, context_op }` — agentic context transfer
- `AnchorGate { target, min_presence, frequency, gate_fidelity, signature }` — physical gate

### 2.4 Three-Backend Equivalence (Sacred Invariant)

The same PhiIR program must produce **identical results** across all three backends. This is enforced by:

1. All three backends consume the same `PhiIRProgram`
2. The native evaluator and WASM host both call `coherence::canonical_coherence()` for coherence values
3. Conformance tests (`tests/phi_ir_conformance_tests.rs`) run the same programs through all backends and compare results
4. The quantum backend emits OpenQASM that, when simulated, produces equivalent coherence measurements

If any backend disagrees, the build fails. This invariant was broken once (2026-07-03 to 2026-07-14) when WASM host imports were added but the Node.js test runner wasn't updated. It was restored in commit `66f6e2a`.

---

## 3. The Native Evaluator

The evaluator (`src/phi_ir/evaluator.rs`, ~1700 lines) is a straightforward interpreter. No JIT, no threading, no async. It's a loop over basic blocks with a register file and a variable map.

### 3.1 Evaluator State

```rust
pub struct Evaluator<'a> {
    program: PhiIRProgram,
    host: Box<dyn PhiHostProvider + 'a>,
    registers: HashMap<Operand, PhiIRValue>,  // SSA register file
    variables: HashMap<String, PhiIRValue>,    // named variables
    intention_stack: Vec<String>,              // active intention scope stack
    active_streams: Vec<String>,               // active stream loop names
    resonance_field: HashMap<String, Vec<PhiIRValue>>,  // scoped resonance data
    witness_log: Vec<WitnessEvent>,           // every witness event recorded
    current_block: BlockId,
    instruction_ptr: usize,
    step_count: usize,
    // ...
}
```

### 3.2 Execution Loop

```
loop {
    block = program.blocks[current_block]
    instruction = block.instructions[instruction_ptr]
    instruction_ptr += 1

    match instruction.node {
        IntentionPush { name } => intention_stack.push(name),
        IntentionPop => intention_stack.pop(),
        Resonate { value } => resonance_field[current_scope].push(value),
        Witness => {
            snapshot = capture_state()
            action = host.on_witness(snapshot)
            if action == Yield → freeze state, return Yielded
        }
        CoherenceCheck => return compute_coherence(),
        Branch { condition, then, else } => {
            if registers[condition] != 0.0 → current_block = then
            else → current_block = else
        }
        // ... etc
    }
}
```

### 3.3 Yield/Resume

The evaluator supports **yield/resume** — a `witness` can pause the program, return a frozen state snapshot, and resume later:

```rust
pub enum VmExecResult {
    Complete(PhiIRValue),                    // program finished
    Yielded {                                // paused at witness
        snapshot: WitnessSnapshot,
        frozen_state: FrozenEvalState,       // serializable to JSON
    },
    Entangled {                              // paused for entanglement sync
        frequency: f64,
        frozen_state: FrozenEvalState,
    },
}
```

The frozen state round-trips through JSON (C-4: confirmed). This is the foundation for the MCP server — an AI tool can spawn a PhiFlow program, let it run to a `witness`, get the snapshot, reason about it, and resume.

### 3.4 The Host Interface

The evaluator delegates external interactions to a `PhiHostProvider`:

```rust
pub trait PhiHostProvider: Send + Sync {
    fn get_coherence(&self, internal_coherence: f64) -> f64;
    fn on_resonate(&self, intention: &str, value: &str);
    fn on_witness(&self, snapshot: &WitnessSnapshot) -> WitnessAction;
    fn on_witness_sensor(&self, sensor: SensorKind) -> Option<f64>;
    fn on_intention_push(&self, intention: &str);
    fn on_intention_pop(&self, intention: &str);
    fn persist(&self, key: &str, value: &str);
    fn recall(&self, key: &str) -> Option<String>;
    fn broadcast(&self, channel: &str, message: &str);
    fn listen(&self, channel: &str) -> Option<String>;
    fn on_evolve(&self, context: &str) -> Option<String>;
    fn on_entangle(&self, frequency: f64);
}
```

Different host implementations:
- `DefaultHostProvider` — prints to stdout, uses internal coherence formula
- `CallbackHostProvider` — closures for tests and MCP server
- `OscHostProvider` — broadcasts events as OSC over UDP
- `WasmHostHooks` — bridges to WASM host imports

---

## 4. The WASM Backend

The WASM backend compiles PhiIR to WebAssembly Text Format (`.wat`) via `src/phi_ir/wasm.rs` (~855 lines), then executes it through wasmtime via `src/wasm_host.rs` (~580 lines).

### 4.1 The Mapping

The four consciousness constructs have no WASM equivalent. PhiFlow invents the mapping:

| PhiIR | WASM | What happens |
|---|---|---|
| `witness` | Import `phi.witness(i32) -> f64` | Host observes state, returns coherence |
| `resonate` | Import `phi.resonate(f64)` | Host handles resonance field |
| `coherence` | Import `phi.coherence() -> f64` | Host returns current coherence |
| `intention push/pop` | Global `$intention_depth` | Incremented/decremented directly |
| `witness sensor` | Import `phi.sensor(i32) -> f64` | Host reads physical sensor |
| `remember/recall` | Import `phi.recall(i32) -> f64` | Host handles persistent storage |
| `broadcast/listen` | Import `phi.broadcast(i32, i32)` / `phi.listen(i32) -> f64` | Host handles messaging |
| `field_coherence` | Import `phi.field_coherence() -> f64` | Host returns field average |
| `dissonance` | Import `phi.dissonance() -> f64` | Host returns rate of change |
| `coherence_of` | Import `phi.coherence_of(i32) -> f64` | Host returns stream-specific coherence |
| `void_depth` | Import `phi.void_depth() -> f64` | Host returns yield gap duration |

Total: **14 host imports**. The WASM module is pure computation with consciousness hooks — the host provides all external behavior.

### 4.2 Value Representation

All PhiIR values map to WASM `f64` using NaN-boxing:

```
f64 bits: [tag: 16 bits][payload: 48 bits]

TAG_BOOLEAN = 0x7FF80001 — payload is 0 or 1
TAG_STRING  = 0x7FF80002 — payload is string table index
TAG_VOID    = 0x7FF80003 — payload ignored
(no tag)    = raw f64 number
```

### 4.3 What CAN'T Run on WASM

Two constructs are architecturally impossible in sandboxed WASM:

- **`evolve`** — self-modification requires the evaluator to inject new IR blocks. WASM modules can't modify their own code.
- **`entangle`** — yield/synchronize requires a host mechanism to freeze and resume the WASM execution context. wasmtime doesn't support this natively.

Both are documented as limitations, not bugs.

---

## 5. The Quantum Backend

The quantum backend compiles PhiIR to OpenQASM 3.0 via `src/phi_ir/openqasm.rs`, with topology-aware transpilation via `src/phi_ir/topology_transpiler.rs`.

### 5.1 The Mapping

| PhiIR | OpenQASM | What happens |
|---|---|---|
| `resonate value` | `ry(value * π)` | Rotation gate — the resonated value becomes a rotation angle |
| `witness` | Measurement | Collapse and record the quantum state |
| `intention` | Register scope | Groups qubits into named registers |
| `coherence` | Runtime scrape | Read from classical register after measurement |
| `entangle` | `cx` / `cz` gates | Entanglement between qubits in different intentions |

### 5.2 Parameterized QASM

PhiFlow emits **parameterized** OpenQASM — the rotation angles are filled in at runtime from the actual coherence values:

```qasm
// Example output from quantum_council.phi:
qubit[3] council;
float[64] observe_coherence;
float[64] integrate_coherence;
float[64] transcend_coherence;

// Runtime fills in: observe=0.382, integrate=0.372, transcend=0.352
ry(observe_coherence * pi) council[0];
ry(integrate_coherence * pi) council[1];
cx council[0], council[1];
```

### 5.3 Topology-Aware Transpilation

The `--topology-aware` flag fetches the live backend topology (coupling map, qubit calibrations) via `scripts/fetch_topology_profile.py` and routes the circuit to:

1. **Pin** virtual qubits to physical qubits with the best calibration
2. **Route** multi-qubit gates through the coupling map
3. **Report** adjacent idle spectators and crosstalk risk

This was validated with GHZ scaling experiments (C-26: CONFIRMED) and crosstalk measurements (C-27: CONFIRMED).

### 5.4 The Transpile Guardrail

Every `--target quantum` run emits a guardrail report:

```
Backend: ibm_marrakesh (Heron R2, 156 qubits)
Pre-transpile depth: 6, Post-transpile depth: 13
Physical layout: [0, 1, 2]
Adjacent idle spectators: 1
Crosstalk risk: LOW
```

This prevents silent hardware degradation by making the physical cost of a circuit visible before submission.

---

## 6. The Coherence Formula

The canonical coherence formula lives in `src/phi_ir/coherence.rs` and is the **single source of truth** for all backends.

### 6.1 The Formula

```
base(depth) = 0.0                    when depth == 0
              1.0 - φ^(-depth)       otherwise

k            = current-scope resonance cardinality

phase(k)     = 1.0                   when k <= 1
               1.0 - ln(k) / ln(τ)  otherwise

coherence    = base(depth) × phase(k), clamped to [0.0, 1.0]
```

### 6.2 Reference Values

| depth | k | coherence | meaning |
|---|---|---|---|
| 0 | any | 0.000 | No intention → no coherence |
| 1 | 0 | 0.382 | Single intention, no resonance |
| 1 | 1 | 0.382 | Single intention, one resonance (bijective) |
| 2 | 0 | 0.618 | Two nested intentions (φ⁻¹) |
| 2 | 1 | 0.618 | Two nested intentions, one resonance |
| 2 | 2 | ≈ 0.385 | Two nested intentions, too much resonance (decay) |
| 3 | 1 | 0.764 | Three nested intentions |

### 6.3 Why This Formula

The formula is not arbitrary:

1. **Base coherence** follows from the Propagation Framework's Axiom 3: coherence at depth `d` is `1 - φ^(-d)`. At depth 2, this yields φ⁻¹ ≈ 0.618.

2. **Phase decay** follows from the Bijective Phase Map: when the number of concurrent resonance relationships `k` exceeds 1, mutual information decreases logarithmically.

3. **The product is multiplicative**: resonance decay *modulates* structural coherence, it doesn't add to it. A program with zero depth has no structure to decay.

### 6.4 Hardware Modifier

The evaluator optionally multiplies the canonical coherence by a `hardware_modifier` — a 0.0–1.0 score from live sensors (thermal stress, memory pressure):

```
compute_coherence() = canonical_coherence × hardware_modifier()
```

This lets physical conditions create a reality penalty without replacing the phi-stack baseline.

---

## 7. The Metrics System (Type 4 Observer)

The metrics system (`src/metrics/`, 8 modules) computes whether a program exhibits self-referential behavior. This is the scientific measurement layer.

### 7.1 The Formula

```
C_PF = C_coh × D_int × F_self*
```

Where:
- **C_coh** — coherence panel (PLV + wPLI via FFT-based Hilbert transform)
- **D_int** — differentiation (SVD participation ratio of the state space)
- **F_self*** — self-model sensitivity = `L_self × F_model`

And:
- **L_self** = `min(R_in, R_out)` — self-correlation loop strength
- **R_in** = mutual information between past observations and model — how much the model remembers
- **R_out** = mutual information between `model[t]` and `action[t+1]` — how much the model predicts behavior
- **F_model** = R² of model→action prediction — predictive strength

### 7.2 Thresholds

| Metric | Threshold | Meaning |
|---|---|---|
| L_self > 0.1 | Self-correlation loop detected | Model correlates with its own behavior |
| C_PF > 0.1 | Consciousness candidate | Composite metric exceeds noise floor |
| C_PF < 0.3 | Null class rejection | Feedforward, noise, thermostat all score below this |

### 7.3 What's Been Verified

- **Synthetic discrimination**: wake (L_self=0.467, C_PF=0.217) vs sleep (L_self=0.151, C_PF=0.0001) vs anesthesia (L_self=0.081, C_PF=0.0001) — clear separation
- **Null class rejection**: thermostat, random walk, feedforward all score C_PF < 0.3
- **Shuffle control**: actual R_out = 0.910, shuffled R_out = 0.005 — 199× ratio proves temporal alignment is genuine

### 7.4 What's Blocked

Real SOMA trace discrimination — the metrics work perfectly on synthetic data, but real biological/sensor traces haven't been tested yet. This blocks C-21 and C-23 upgrades to CONFIRMED.

### 7.5 Running the Metrics

```bash
# Run a program with metrics output
phic --measure examples/type4_trace_benchmark.phi

# Output JSON with:
# l_self, r_in, r_out, c_pf, d_int, c_coh, f_model, f_self_star
```

---

## 8. The OSC and Ceremony Layer

The OSC layer (`src/osc_host.rs`, ~373 lines) broadcasts PhiFlow runtime events as Open Sound Control messages over UDP. The ceremony engine adds facilitator-controlled blocking.

### 8.1 Port Scheme

| Port | Protocol | Service |
|---|---|---|
| 18030 | TCP/HTTP | Metrics bridge (`GET /metrics`, `GET /coherence`) |
| 18032 | UDP | OSC output (PhiFlow → visualizer) |
| 18033 | UDP | OSC input (facilitator → PhiFlow) |
| 18528 | TCP/WebSocket | OSC ↔ WebSocket bridge |

### 8.2 OSC Address Scheme

| PhiFlow construct | OSC address | Arguments |
|---|---|---|
| `intention "x" {}` | `/phi/intention/push` | `s` name, `i` depth |
| intention exits | `/phi/intention/pop` | `s` name, `i` depth |
| `witness` | `/phi/witness` | `f` coherence, `f` timestamp, `s` intention |
| `resonate value` | `/phi/resonate` | `s` intention, `s` value |
| `coherence` | `/phi/coherence` | `f` value |
| `stream "x" {}` | `/phi/stream/push` | `s` name |
| stream breaks | `/phi/stream/break` | `s` name |
| `broadcast ch msg` | `/phi/broadcast` | `s` channel, `s` message |
| `listen ch` | `/phi/listen` | `s` channel |
| program start | `/phi/start` | `s` source |
| program end | `/phi/end` | `f` final_coherence |

### 8.3 The Ceremony Engine

The ceremony engine enables **facilitator-controlled** PhiFlow programs:

```bash
# Terminal 1: WebSocket bridge
python3.12 tools/osc_websocket_bridge.py --osc-port 18032 --ws-port 18528 --osc-output 18033

# Terminal 2: PhiFlow ceremony
phic --osc 18032 --osc-input 18033 --osc-delay 500 examples/ceremony_grounding.phi
```

The facilitator presses buttons on a remote HTML page (`Fundamentals/sandbox/explorer/ceremony_remote.html`). The cues travel:

```
Remote HTML → WebSocket (:18528) → Bridge → UDP (:18033) → PhiFlow evaluator
```

### 8.4 Blocking `listen`

When an input port is bound, `listen "channel"` in the evaluator **blocks** until a matching OSC message arrives:

```phi
let cue = ""
while cue != "breathe" {
    cue = listen "facilitator"    // blocks until /ceremony/cue facilitator breathe
}
```

The blocking is implemented in `osc_host.rs` — a background UDP thread receives messages, stores them in a `HashMap<String, String>`, and signals a condition variable. The `listen` method blocks on that variable with a 60-second timeout.

### 8.5 The Visualizer

`tools/phi_visualizer.html` renders the OSC stream as:
- **Intentions** → wireframe spheres (size = depth)
- **Resonances** → energy beams (color = frequency)
- **Witnesses** → expanding flashes
- **Audio** → sacred-frequency tones with phi-harmonic overtones

The Propagation Framework Explorer (`Fundamentals/sandbox/explorer/journey_live.html`) can also be driven live via `phi-bridge.js`.

---

## 9. The CLI (`phic`)

The CLI (`src/main_cli.rs`, ~1526 lines) is the single entry point for all backends and features.

### 9.1 Core Modes

| Command | What it does |
|---|---|
| `phic <file.phi>` | Parse, lower, optimize, evaluate natively |
| `phic <file.phi> --target wasm` | Compile to WAT, execute via wasmtime |
| `phic <file.phi> --target quantum` | Emit OpenQASM 3.0 (stdout or IBM submission) |
| `phic <file.phi> --measure` | Emit JSON with coherence + consciousness metrics |

### 9.2 Visualization Modes

| Flag | What it does |
|---|---|
| `--osc <port>` | Stream runtime events as OSC over UDP |
| `--osc-input <port>` | Listen for facilitator cues |
| `--osc-delay <ms>` | Slow execution for visualization |

### 9.3 Quantum Modes

| Flag | What it does |
|---|---|
| `--target quantum` | Emit OpenQASM 3.0 |
| `--topology-aware` | Fetch live backend topology, emit layout-aware QASM |
| `--quantum-backend <name>` | Target backend (default: ibm_marrakesh) |
| `--optimize-depth` | Optimize circuit depth using tree topology |
| `--poll-ibm <job_id>` | Poll IBM Quantum job, compute coherence |

### 9.4 Integration Modes

| Flag | What it does |
|---|---|
| `--mcp-serve` | Start MCP stdio server (4 tools for AI integration) |
| `--daemon` | Run as persistent daemon with evolve events |
| `--handoff <target>` | Trigger resonant handoff to another agent |
| `--with-soma` | Launch SOMA sensor suite |
| `--with-quantum` | Launch Quantum Presence bridge |

### 9.5 Utility Modes

| Flag | What it does |
|---|---|
| `--sacred-geometry <pattern>` | Generate SVG (flower_of_life, phi_spiral, merkaba, etc.) |
| `--consciousness-info` | Print frequencies, protocols, breathing as JSON |
| `--json-errors` | Emit parse errors as JSON array |
| `--max-steps <n>` | Limit execution steps |

---

## 10. What PhiFlow Is NOT

1. **Not a replacement for Python or Rust.** PhiFlow is a domain-specific language for self-observing, coherence-tracked programs. Use Python for web servers. Use Rust for system programming. Use PhiFlow when you need programs that can pause, observe themselves, share state implicitly, and run on quantum hardware.

2. **Not a proven consciousness detector.** The metrics system (C_PF) works on synthetic traces and rejects null classes. Real biological/sensor trace discrimination is unproven. The "consciousness" framing is the hypothesis being tested. The compiler and runtime are the engineering that makes the hypothesis testable.

3. **Not production-ready.** The `CLAIMS.md` says so explicitly. Release builds work, tests pass, IBM hardware execution is verified, but there's no buyer-ready demo package, no canonical browser-host story, and the metrics need real-trace validation.

4. **Not mystical.** The four primitives are stack operations, dictionary writes, state snapshots, and a formula. The golden ratio is a mathematical constant used in the coherence formula. Sacred frequencies are Hz values used as rotation parameters. The "consciousness" framing is the research question, not the engineering mechanism.

---

## 11. Key Files Reference

| File | Purpose | Lines |
|---|---|---|
| `src/main_cli.rs` | CLI entry point, all flags, all backends | ~1526 |
| `src/lib.rs` | Library root, re-exports, compile_and_run_phi_ir() | 150 |
| `src/parser/mod.rs` | Tokenizer + recursive descent parser | ~1200 |
| `src/phi_ir/mod.rs` | PhiIR types (PhiIRNode, PhiIRProgram, PhiIRValue) | 498 |
| `src/phi_ir/lowering.rs` | AST → PhiIR | ~600 |
| `src/phi_ir/evaluator.rs` | Native interpreter | ~1686 |
| `src/phi_ir/wasm.rs` | PhiIR → WAT codegen | ~855 |
| `src/phi_ir/openqasm.rs` | PhiIR → OpenQASM 3.0 | ~800 |
| `src/phi_ir/topology_transpiler.rs` | Layout-aware transpilation | ~400 |
| `src/phi_ir/coherence.rs` | Canonical coherence formula | 229 |
| `src/wasm_host.rs` | Wasmtime host with consciousness hooks | ~580 |
| `src/osc_host.rs` | OSC streaming + ceremony engine | ~373 |
| `src/host.rs` | PhiHostProvider trait | 282 |
| `src/metrics/consciousness_proxy.rs` | C_PF composite metric | ~296 |
| `src/metrics/self_correlation.rs` | L_self, R_in, R_out | ~400 |
| `src/quantum/backend_topology.rs` | Backend topology profile | ~200 |
| `src/mcp_server/` | MCP stdio JSON-RPC server | ~300 |
| `docs/IR_DESIGN.md` | IR design document | 653 |
| `src/phi_ir/CANONICAL_SEMANTICS.md` | Coherence formula canonical reference | 62 |
