# PhiFlow Intermediate Representation (IR) Design

**Status**: Proposal
**Date**: 2026-02-12
**Authors**: Synthesized from WASM, Quantum, and Hardware backend proposals

---

## 1. Why PhiFlow Needs an IR

### The Problem

Today, PhiFlow has one execution path:

```
.phi file --> Lexer --> Parser --> AST (PhiExpression) --> Tree-Walking Interpreter --> Output
```

The interpreter walks the AST directly. This works, but it locks PhiFlow into a single execution model. There is no way to compile a `.phi` program to run in a browser, on a quantum computer, or on an ESP32 microcontroller without rewriting the interpreter for each target.

### What an IR Is

An Intermediate Representation is a data structure that sits between the AST and the backends. Think of it like a common language that all backends can read.

If you know LLVM: LLVM IR is what Clang (C/C++), Rustc (Rust), and Swift all compile *to*, and then LLVM compiles *from* that IR to x86, ARM, WASM, etc. The frontends don't know about the backends. The backends don't know about the frontends. The IR is the contract between them.

PhiFlow needs the same thing, but with a twist: PhiFlow's four unique constructs (`witness`, `intention`, `resonate`, and live coherence) have no equivalent in any existing IR. They must be preserved as first-class citizens in the IR, not lowered into generic function calls, because each backend interprets them in fundamentally different ways.

### The Target Pipeline

```
.phi file --> Lexer --> Parser --> AST (PhiExpression)
                                        |
                                        v
                                   PhiIR (shared)
                                   /    |    \
                                  v     v     v
                               WASM  Quantum  Hardware
                               (JS)  (QASM)   (ESP32)
```

One shared IR. Three backends. Each backend lowers the same IR nodes into its own execution model.

---

## 2. The Two-Layer Architecture

All three backend teams converged on the same structural insight: the IR should have two layers.

### Layer 1: PhiIR (Shared)

A single Rust enum that every backend consumes. This layer preserves:

- All four PhiFlow-unique constructs as dedicated node types
- Standard computation (arithmetic, variables, control flow, functions)
- Pattern creation as a dedicated node (not a generic call)
- Type information sufficient for all backends

This layer is **backend-agnostic**. It does not mention WASM linear memory, qubit registers, or ESP32 peripherals.

### Layer 2: Backend-Specific Lowering

Each backend transforms PhiIR nodes into its own representation:

| Backend | Layer 2 Output | What It Looks Like |
|---------|---------------|-------------------|
| **WASM** | WASM module bytes | SSA-style instructions, host function imports, tagged PhiValue in linear memory |
| **Quantum** | Circuit description | Named quantum registers, gate sequences, entanglement patterns, measurement operations |
| **Hardware** | Compact bytecode | Fixed 4-byte instructions, f32 arithmetic, HAL trait calls, MQTT operations |

The key principle: **Layer 1 says WHAT to do. Layer 2 says HOW to do it on a specific target.**

---

## 3. PhiFlow's Four Unique Constructs in the IR

This is the heart of the design. Each of PhiFlow's four constructs must be a first-class IR node because each backend interprets it in a fundamentally different way.

### 3.1 `witness` -- Self-Observation

**What it means**: The program pauses to observe its own state. Not a print statement. Not a breakpoint. The program becomes present with its own computation.

**IR Node**: `Witness { target: Option<Operand> }`

| Backend | Interpretation |
|---------|---------------|
| **WASM** | Call JS host function `__phi_witness(value)`. The host captures current coherence, intention stack, frequencies used, and presents them to the user. Coherence increases (self-observation is alignment-positive). |
| **Quantum** | **Partial measurement**. If `target` is a qubit reference, perform mid-circuit measurement with a collapse policy (mid-circuit, deferred, or non-destructive depending on context). If bare `witness`, measure an ancilla qubit that reflects circuit fidelity without collapsing the main computation. |
| **Hardware** | **Sensor read**. Bare `witness` reads all active sensors (EEG, HRV, temperature, microphone) into a snapshot struct. `witness expression` reads the sensor most relevant to the expression's frequency. Data goes into a fixed-capacity ring buffer, not heap-allocated. |

### 3.2 `intention` -- Purpose Before Process

**What it means**: The program declares WHY before HOW. An `intention "healing" { ... }` block executes the same operations as the code outside it, but the coherence calculation accounts for whether operations align with the declared purpose.

**IR Nodes**: `IntentionPush { name: String }` and `IntentionPop`

The parser's `IntentionBlock` is expanded into a push/pop pair wrapping the body:

```
IntentionPush { name: "healing" }
CoherenceCheck                       // coherence snapshot on entry
... body instructions ...
CoherenceCheck                       // coherence snapshot on exit
IntentionPop
```

| Backend | Interpretation |
|---------|---------------|
| **WASM** | Push intention name onto a runtime stack in linear memory. All subsequent `witness` and `resonate` operations reference this name. Pop on exit. JS host tracks the intention stack for reporting. |
| **Quantum** | **Allocate a named quantum register**. Each intention gets its own register (e.g., `healing[0..3]`). Qubit references within the block are always `(register_name, index)`, never flat global indices. This keeps intention blocks isolated until explicitly entangled via `resonate`. |
| **Hardware** | **Reconfigure sensor pipeline**. Different intentions activate different sensor configurations. `"healing"` might prioritize HRV and EEG alpha waves. `"analysis"` might prioritize EEG gamma and keyboard activity. The intention name indexes into a compile-time lookup table of sensor configs. |

### 3.3 `resonate` -- Internal Communication

**What it means**: Intention blocks share state through a resonance field. Not function calls. Not message passing. Values deposited by one intention become visible to others.

**IR Node**: `Resonate { value: Option<Operand> }`

| Backend | Interpretation |
|---------|---------------|
| **WASM** | Call JS host function `__phi_resonate(intention_name, value)`. The JS host maintains the resonance field (a `Map<string, Array>`) and makes deposited values available to subsequent intention blocks. Bare `resonate` shares all variables in the current scope. |
| **Quantum** | **Entanglement**. Apply entangling gates (CNOT, CZ) between the current intention's register and all registers that have previously resonated. The coupling strength is scaled by the phi-harmonic relationship between the two intentions' frequencies. A `FrequencyRelationship` annotation on the IR node tells the backend what ratio to use. |
| **Hardware** | **MQTT publish**. `resonate value` serializes the value and publishes it to topic `phi/{device_id}/{intention_name}`. Other P1 devices subscribed to that topic receive the resonance. Bare `resonate` publishes a snapshot of all sensor readings. Uses fixed-size message buffers (64 bytes max per publish). |

### 3.4 Live Coherence -- Self-Measurement

**What it means**: The program continuously measures its own alignment on a 0.0 to 1.0 scale. Coherence starts at 1.0 and changes based on whether operations align with declared intentions and sacred frequency relationships.

**IR Node**: `CoherenceCheck`

This node is a **hook point**, not an implementation. It tells the backend: "evaluate coherence NOW using whatever method is appropriate for this target."

| Backend | Interpretation |
|---------|---------------|
| **WASM** | Pure math in WASM memory. The module maintains a coherence float and a frequency history array. `CoherenceCheck` runs a function that examines frequencies used, intention alignment, witness count, and contradictions to compute the current score. No host call needed -- this is deterministic computation. |
| **Quantum** | **State fidelity**. Compare the current quantum state against the ideal state for the active intention. This may involve computing the fidelity metric `F(rho, sigma)` or, more practically, tracking how many non-phi-harmonic gates have been applied vs. phi-harmonic ones. |
| **Hardware** | **Biometric coherence**. Read from the HAL trait's `compute_coherence()` method, which combines real sensor data: HRV coherence (heart rhythm regularity), EEG alpha/theta coupling, and physiological alignment metrics. The result is a real number from real human biology, not frequency math. |

---

## 4. Shared IR Node Types

The PhiIR enum contains three categories of nodes.

### 4.1 Standard Computation

These are conventional and appear in any IR:

| Node | Purpose |
|------|---------|
| `Const(PhiIRValue)` | Literal number, string, boolean |
| `LoadVar(String)` | Read a variable |
| `StoreVar(String, Operand)` | Write a variable |
| `BinOp(BinOp, Operand, Operand)` | Arithmetic, comparison, logic |
| `UnaryOp(UnOp, Operand)` | Negation, boolean not |
| `Call(String, Vec<Operand>)` | Function call |
| `Return(Operand)` | Return from function |
| `Branch(Operand, BlockId, BlockId)` | Conditional jump (if/else) |
| `Jump(BlockId)` | Unconditional jump (loop back-edge) |
| `ListNew(Vec<Operand>)` | Create a list |
| `ListGet(Operand, Operand)` | Index into a list |
| `FuncDef(String, Vec<Param>, BlockId)` | Function definition |

### 4.2 PhiFlow-Unique Nodes

These are the four constructs plus pattern creation. They MUST remain as dedicated nodes because each backend interprets them differently:

| Node | Purpose | Why It Cannot Be Lowered |
|------|---------|-------------------------|
| `Witness(Option<Operand>)` | Self-observation | WASM: host call. Quantum: measurement. Hardware: sensor read. |
| `IntentionPush(String)` | Enter intention scope | WASM: stack push. Quantum: register allocation. Hardware: sensor reconfig. |
| `IntentionPop` | Exit intention scope | Paired with push. |
| `Resonate(Option<Operand>)` | Share state between intentions | WASM: host map. Quantum: entanglement. Hardware: MQTT. |
| `CoherenceCheck` | Evaluate program alignment | WASM: pure math. Quantum: fidelity. Hardware: biometric. |
| `CreatePattern(PatternKind, Operand, HashMap<String, Operand>)` | Sacred geometry generation | WASM: compute points. Quantum: initialize register state. Hardware: DAC output. |

### 4.3 Domain Operations

These are specialized operations from the AST that get lowered to typed host/HAL calls:

| Node | Purpose |
|------|---------|
| `DomainCall(DomainOp, Vec<Operand>)` | Covers AudioSynthesis, ConsciousnessMonitor, HardwareSync, EmergencyProtocol, BiologicalInterface, QuantumField, FrequencyPattern, ConsciousnessFlow |

Each backend maps `DomainOp` variants to its own implementation. Backends that don't support a domain operation can emit a compile-time warning or a no-op.

---

## 5. What Each Backend Needs

### 5.1 WASM Backend

**Execution model**: Compiled WASM module + JavaScript host runtime.

**Key requirements from the IR**:
- **SSA-style flat instructions**: The IR should be a flat list of instructions within basic blocks, not a tree. This maps naturally to WASM's stack machine after a straightforward linearization pass.
- **Tagged PhiValue**: All runtime values are a 9-byte tagged union (1 byte tag + 8 bytes payload). Strings and patterns are stored in linear memory with pointer+length.
- **Host function boundary**: The IR must clearly distinguish what runs in WASM (pure computation, coherence math) from what calls out to JS (witness display, resonance field management, audio output, visualization).
- **Resonance field in JS host**: The resonance `Map<string, Array>` lives in JavaScript, not in WASM linear memory. `Resonate` nodes compile to imported host function calls.

**What WASM does NOT need from the IR**:
- Qubit references or quantum register information
- Sensor HAL trait definitions
- MQTT topic structures
- `no_std` constraints

### 5.2 Quantum Backend

**Execution model**: Classical controller orchestrating quantum circuit operations.

**Key requirements from the IR**:
- **Classical vs. quantum separation**: The IR must let the quantum backend identify which sections are classical scaffolding (variable setup, control flow, coherence math) and which are quantum operations (pattern creation as state preparation, witness as measurement, resonate as entanglement).
- **Named registers per intention**: Qubit references are always `(register_name: String, index: u32)`, never flat global indices. The intention name IS the register name.
- **Frequency as rotation angle**: Sacred frequencies must be preserved in the IR (not pre-computed to generic floats) so the quantum backend can compute rotation angles as `freq / 432.0 * PI / 2.0`. The IR should store both the original frequency value and a `SacredFrequency` enum variant when applicable.
- **FrequencyRelationship annotations**: When two intention blocks resonate, the IR should annotate the relationship between their frequencies (e.g., `PhiHarmonic(528/432)` = ratio of 1.222...). The quantum backend uses this to determine entanglement coupling strength.
- **Collapse policy on witness**: The IR's `Witness` node should carry a hint about measurement strategy (`MidCircuit`, `Deferred`, `NonDestructive`). The quantum backend uses this; other backends ignore it.

**What quantum does NOT need from the IR**:
- WASM linear memory layout
- ESP32 peripheral addresses
- MQTT topics
- Emergency protocol interrupt priorities

### 5.3 Hardware Backend (ESP32)

**Execution model**: Compact bytecode VM running on ESP32 with real-time sensor integration.

**Key requirements from the IR**:
- **f32 not f64**: ESP32 has no FPU64. The IR uses f64 (for precision in WASM and quantum), but the hardware backend must downcast to f32 during lowering. The IR should not bake in f64-specific assumptions (like epsilon values for comparison).
- **Fixed-size everything**: No heap allocation. The IR should not require backends to allocate variable-sized structures at runtime. Strings interned at compile time. Ring buffers instead of growable vectors. Max 32KB bytecode + 8KB constants per program.
- **HAL trait for peripherals**: All sensor access, MQTT, DAC output, and LED control goes through a Hardware Abstraction Layer trait. The IR's domain operations map to HAL methods, not direct peripheral register writes.
- **Interrupt-driven emergency protocols**: `EmergencyProtocol` IR nodes must be flagged as interrupt-priority. On hardware, these preempt normal execution with <100ms response time for seizure detection.
- **Procedural pattern generation**: `CreatePattern` cannot store point arrays (520KB total SRAM). Instead, the hardware backend compiles patterns to procedural DAC output functions that generate points on-the-fly.

**What hardware does NOT need from the IR**:
- Qubit references or quantum registers
- JS host function imports
- Full f64 precision
- Heap-allocated collections

---

## 6. Tensions and Resolutions

The three backends disagree on several fundamental points. Here is how each tension resolves.

### 6.1 Numeric Precision: f64 vs f32 vs Angles

**Tension**: WASM uses f64. Hardware needs f32. Quantum needs angles derived from frequency ratios.

**Resolution**: The shared IR uses f64 for all numeric values. Each backend is responsible for its own precision lowering:
- WASM: uses f64 directly (native WASM type)
- Hardware: downcasts to f32 during bytecode emission, with explicit rounding strategy
- Quantum: computes angles from frequency values during circuit generation

The IR also preserves `SacredFrequency` annotations on frequency-bearing nodes. This costs nothing for backends that ignore it, and gives quantum the symbolic information it needs.

### 6.2 IR Granularity: SSA vs Circuits vs Bytecode

**Tension**: WASM wants SSA-style basic blocks. Quantum wants circuit sections grouped by intention. Hardware wants a flat instruction stream.

**Resolution**: The shared IR uses **basic blocks** (a list of named blocks, each containing a sequence of IR nodes and a terminator). This is the most flexible common denominator:
- WASM linearizes blocks into its stack machine
- Quantum groups blocks by intention register for circuit construction
- Hardware flattens blocks into a sequential bytecode stream

Basic blocks are the standard IR unit in compiler design (LLVM, Cranelift, GCC) because they're simple to analyze and transform, and every execution model can consume them.

### 6.3 Quantum's Fundamentally Different Execution Model

**Tension**: Classical backends (WASM, hardware) execute instructions sequentially. Quantum backends prepare states, apply gates, and measure. These are fundamentally different computational models.

**Resolution**: The shared IR does NOT try to encode quantum gates. Instead:
1. The quantum backend receives the same PhiIR as everyone else
2. It interprets `IntentionPush` as "allocate a quantum register"
3. It interprets `CreatePattern` as "prepare an initial quantum state"
4. It interprets `Resonate` as "apply entangling gates"
5. It interprets `Witness` as "perform measurement"
6. It interprets `CoherenceCheck` as "compute state fidelity"
7. Standard computation nodes (arithmetic, variables, control flow) become **classical controller code** that orchestrates the quantum operations

The PhiIR nodes are semantic hooks. The quantum backend maps them to quantum operations. The classical backends map them to classical operations. The IR does not need to know about either model.

### 6.4 CreatePattern: First-Class Node vs Lowered Instruction

**Tension**: WASM and quantum want `CreatePattern` as a first-class IR node that they interpret in their own way. Hardware wants to lower it to `OP_FREQ_SET` + `OP_PATTERN_GEN` bytecodes.

**Resolution**: `CreatePattern` stays as a first-class IR node in the shared PhiIR. Hardware's bytecode emitter lowers it to its own opcodes during Layer 2 lowering. This is exactly what the two-layer architecture is for -- the shared IR preserves semantic meaning, and each backend lowers it as needed.

### 6.5 Coherence Ownership: Computed vs Delegated vs Fidelity

**Tension**: WASM computes coherence from frequency history (pure math). Hardware delegates to a firmware HAL (real biometrics). Quantum measures state fidelity.

**Resolution**: `CoherenceCheck` is a **hook point** in the IR, not an implementation. It says "evaluate coherence now." How that evaluation happens is entirely the backend's decision:
- The IR carries enough context (current intention, frequency history, witness count) for any backend to make its own decision
- WASM implements coherence as a pure function over that context
- Hardware calls `hal.compute_coherence()` which reads real sensors
- Quantum computes fidelity against an ideal state

No single coherence algorithm is baked into the IR.

---

## 7. Proposed Unified IR

### 7.1 Core Types

```rust
/// A reference to a computed value (SSA-style)
type Operand = u32;  // index into the instruction list

/// Basic block identifier
type BlockId = u32;

/// Sacred frequency annotation (optional, backends can ignore)
enum SacredFrequency {
    Ground,        // 432 Hz
    Creation,      // 528 Hz
    Heart,         // 594 Hz
    Voice,         // 672 Hz
    Vision,        // 720 Hz
    Unity,         // 768 Hz
    Source,         // 963 Hz
    Arbitrary(f64), // non-sacred frequency
}

/// Pattern types that CreatePattern can produce
enum PatternKind {
    Spiral, Flower, DNA, Mandelbrot, Pentagram,
    SriYantra, Golden, Fibonacci, Heart, Toroid, Field,
}

/// Measurement strategy hint (quantum backend uses this, others ignore)
enum CollapsePolicy {
    MidCircuit,     // measure and continue
    Deferred,       // record but don't collapse until end
    NonDestructive, // measure ancilla, preserve main state
}

/// Domain operations (specialized features beyond the four core constructs)
enum DomainOp {
    AudioSynthesize,
    ConsciousnessMonitor,
    ConsciousnessState,
    FrequencyPattern,
    QuantumField,
    BiologicalInterface,
    HardwareSync,
    EmergencyProtocol { interrupt_priority: bool },
    ConsciousnessFlow,
    Validate,
}

/// Binary operators
enum PhiIRBinOp {
    Add, Sub, Mul, Div, Mod, Pow,
    Eq, Neq, Lt, Lte, Gt, Gte,
    And, Or,
}

/// Unary operators
enum PhiIRUnOp {
    Neg, Not,
}

/// Function parameter
struct Param {
    name: String,
    // Type annotation is optional in PhiFlow
}
```

### 7.2 The PhiIR Node Enum

```rust
/// One instruction in the PhiIR.
/// Each instruction produces at most one value, referenced by its index (Operand).
enum PhiIRNode {
    // --- Standard Computation ---

    /// Load a constant value
    Const(PhiIRValue),

    /// Read a variable from the environment
    LoadVar(String),

    /// Write a value to a variable
    StoreVar { name: String, value: Operand },

    /// Binary operation
    BinOp { op: PhiIRBinOp, left: Operand, right: Operand },

    /// Unary operation
    UnaryOp { op: PhiIRUnOp, operand: Operand },

    /// Call a user-defined or built-in function
    Call { name: String, args: Vec<Operand> },

    /// Return a value from a function
    Return(Operand),

    /// Create a list
    ListNew(Vec<Operand>),

    /// Index into a list
    ListGet { list: Operand, index: Operand },

    /// Define a function (name, parameters, entry block)
    FuncDef { name: String, params: Vec<Param>, body: BlockId },


    // --- PhiFlow-Unique Nodes (first-class, never lowered in shared IR) ---

    /// Self-observation. The program pauses to observe its own state.
    /// target: None = observe everything. Some(op) = observe a specific value.
    /// collapse_policy: hint for quantum backend (others ignore).
    Witness {
        target: Option<Operand>,
        collapse_policy: CollapsePolicy,
    },

    /// Enter an intention scope. Pushes intention name onto the stack.
    /// WASM: runtime stack. Quantum: register allocation. Hardware: sensor reconfig.
    IntentionPush {
        name: String,
        frequency_hint: Option<SacredFrequency>,
    },

    /// Exit an intention scope. Pops the current intention.
    IntentionPop,

    /// Share state between intention blocks through resonance.
    /// value: None = share all current scope. Some(op) = share specific value.
    Resonate {
        value: Option<Operand>,
        frequency_relationship: Option<f64>,  // phi-harmonic ratio, e.g. 528/432
    },

    /// Evaluate program coherence NOW using backend-appropriate method.
    CoherenceCheck,

    /// Create a sacred geometry pattern at a given frequency.
    CreatePattern {
        kind: PatternKind,
        frequency: Operand,
        frequency_annotation: SacredFrequency,
        params: Vec<(String, Operand)>,
    },


    // --- Domain Operations (backend-specific interpretation) ---

    /// Specialized operation. Each backend maps these to its own implementation.
    /// Backends that don't support a domain op emit a warning or no-op.
    DomainCall {
        op: DomainOp,
        args: Vec<Operand>,
        string_args: Vec<String>,  // for intention names, device types, etc.
    },


    // --- Control Flow (block terminators) ---

    /// Conditional branch
    Branch { condition: Operand, then_block: BlockId, else_block: BlockId },

    /// Unconditional jump
    Jump(BlockId),

    /// Block terminator: fall through (last instruction in block, no explicit jump)
    Fallthrough,
}
```

### 7.3 Program Structure

```rust
/// A basic block: a named sequence of instructions with a terminator.
struct PhiIRBlock {
    id: BlockId,
    label: String,         // human-readable name (e.g., "intention_healing_entry")
    instructions: Vec<PhiIRNode>,
    terminator: PhiIRNode, // must be Branch, Jump, Return, or Fallthrough
}

/// A complete PhiIR program.
struct PhiIRProgram {
    /// All basic blocks in the program, in order.
    blocks: Vec<PhiIRBlock>,

    /// Entry block ID.
    entry: BlockId,

    /// Interned string table (hardware backend needs this for no-heap operation).
    string_table: Vec<String>,

    /// Sacred frequency declarations found in the program.
    frequencies_declared: Vec<(SacredFrequency, f64)>,

    /// Intention names declared in the program (for register pre-allocation by quantum).
    intentions_declared: Vec<String>,
}
```

### 7.4 IR Values

```rust
/// Values that can appear as constants in the IR.
/// Backends lower these to their own representations.
enum PhiIRValue {
    Number(f64),       // WASM: f64, Hardware: f32, Quantum: f64
    String(u32),       // index into string_table
    Boolean(bool),
    Void,
}
```

---

## 8. Worked Example

To make the IR concrete, here is how `code_that_resonates.phi` would compile through the pipeline.

### Source (simplified)

```phi
intention "healing" {
    let pattern = create spiral at 432Hz with { rotations: 13.0, scale: 100.0 }
    witness pattern
    resonate pattern
}

intention "analysis" {
    witness
}
```

### PhiIR Output

```
Block 0 "entry":
    Jump(Block 1)

Block 1 "intention_healing_entry":
    [0] IntentionPush { name: "healing", frequency_hint: Some(Ground) }
    [1] CoherenceCheck
    [2] Const(Number(432.0))
    [3] Const(Number(13.0))
    [4] Const(Number(100.0))
    [5] CreatePattern { kind: Spiral, frequency: @2, annotation: Ground, params: [("rotations", @3), ("scale", @4)] }
    [6] StoreVar { name: "pattern", value: @5 }
    [7] LoadVar("pattern")
    [8] Witness { target: Some(@7), collapse_policy: MidCircuit }
    [9] LoadVar("pattern")
    [10] Resonate { value: Some(@9), frequency_relationship: None }
    [11] CoherenceCheck
    [12] IntentionPop
    Jump(Block 2)

Block 2 "intention_analysis_entry":
    [0] IntentionPush { name: "analysis", frequency_hint: None }
    [1] CoherenceCheck
    [2] Witness { target: None, collapse_policy: MidCircuit }
    [3] CoherenceCheck
    [4] IntentionPop
    Fallthrough
```

### How Each Backend Reads This

**WASM**: Blocks 1 and 2 become linear WASM instructions. `CreatePattern` calls a WASM function that generates spiral points in linear memory. `Witness` calls imported JS host function. `Resonate` calls imported JS host function that stores the value in a JavaScript Map. `CoherenceCheck` calls an in-module function that reads the frequency history array.

**Quantum**: Block 1 allocates register `healing[0..N]`. `CreatePattern` at 432Hz prepares the initial state with rotation `432/432 * pi/2 = pi/2`. `Witness` performs a mid-circuit measurement on an ancilla. `Resonate` marks `healing` as available for entanglement. Block 2 allocates register `analysis[0..M]`. The bare `Witness` measures ancilla reflecting overall fidelity. If `analysis` were to read a resonated value, entangling gates would be applied between the `healing` and `analysis` registers.

**Hardware**: Block 1 emits `OP_INTENT_PUSH "healing"` (reconfigures sensors for healing mode). `CreatePattern` emits `OP_FREQ_SET 432.0` + `OP_PATTERN_GEN SPIRAL` which drives the DAC procedurally. `Witness` emits `OP_SENSOR_READ ALL` into a ring buffer. `Resonate` emits `OP_MQTT_PUB "phi/device/healing"` with the pattern data (truncated to fit 64-byte message). `CoherenceCheck` emits `OP_HAL_COHERENCE` which reads from the biometric HAL.

---

## 9. AST-to-IR Lowering Rules

The conversion from `PhiExpression` (AST) to `PhiIRNode` follows these rules:

| AST Node | IR Node(s) |
|----------|-----------|
| `Number(n)` | `Const(Number(n))` |
| `String(s)` | `Const(String(intern(s)))` |
| `Boolean(b)` | `Const(Boolean(b))` |
| `Variable(name)` | `LoadVar(name)` |
| `LetBinding { name, value, .. }` | Lower `value` to get operand, then `StoreVar { name, value: operand }` |
| `BinaryOp { left, op, right }` | Lower both sides, then `BinOp { op, left, right }` |
| `UnaryOp { op, operand }` | Lower operand, then `UnaryOp { op, operand }` |
| `FunctionDef { name, params, body, .. }` | `FuncDef { name, params, body: new_block(body) }` |
| `FunctionCall { name, args }` | Lower all args, then `Call { name, args }` |
| `Return(expr)` | Lower expr, then `Return(operand)` |
| `IfElse { cond, then, else }` | Lower cond, emit `Branch(cond, then_block, else_block)`, create two new blocks |
| `ForLoop { var, iterable, body }` | Lower to loop header block + body block + exit block with Jump back-edges |
| `WhileLoop { cond, body }` | Lower to loop header block + body block + exit block |
| `Block(exprs)` | Lower each expression in sequence within current block |
| `List(elements)` | Lower each element, then `ListNew(operands)` |
| `ListAccess { list, index }` | Lower both, then `ListGet { list, index }` |
| `Witness { expression, body }` | `Witness { target, collapse_policy: MidCircuit }` (body, if present, lowered as subsequent instructions) |
| `IntentionBlock { intention, body }` | `IntentionPush` + `CoherenceCheck` + lower body + `CoherenceCheck` + `IntentionPop` |
| `Resonate { expression }` | Lower expression if present, then `Resonate { value }` |
| `CreatePattern { pattern_type, frequency, params }` | Lower frequency and all param values, then `CreatePattern { kind, frequency, annotation, params }` |
| `ConsciousnessValidation { .. }` | `DomainCall { op: Validate, .. }` |
| `AudioSynthesis { .. }` | `DomainCall { op: AudioSynthesize, .. }` |
| `EmergencyProtocol { .. }` | `DomainCall { op: EmergencyProtocol { interrupt_priority: true }, .. }` |
| All other AST nodes | `DomainCall { op: <appropriate variant>, .. }` |

---

## 10. Next Steps

### Immediate (to build the IR)

1. **Define `phi_ir` module** in `src/phi_ir/mod.rs` with the types from Section 7
2. **Implement AST-to-IR lowering** in `src/phi_ir/lower.rs` following the rules in Section 9
3. **Add IR pretty-printer** so developers can inspect the IR (critical for debugging backends)
4. **Keep the interpreter working** -- the existing AST interpreter remains the reference implementation; the IR is an alternative compilation path, not a replacement (yet)

### Per-Backend (after shared IR exists)

5. **WASM backend**: Implement `PhiIRProgram -> wasm_module_bytes` in `src/backends/wasm/`
6. **Quantum backend**: Implement `PhiIRProgram -> QuantumCircuit` in `src/backends/quantum/`
7. **Hardware backend**: Implement `PhiIRProgram -> ESP32Bytecode` in `src/backends/hardware/`

### Validation

8. **Round-trip test**: For every example `.phi` program, verify that `AST -> IR -> Interpreter` produces the same output as `AST -> Interpreter` (the existing path)
9. **Backend smoke tests**: Each backend should compile and run at least `code_that_resonates.phi` on its target platform

### Open Questions

- **Should the IR support optimization passes?** (e.g., constant folding, dead code elimination) Not needed immediately, but the basic-block structure supports it when the time comes.
- **How does the IR handle the resonance field across compilation units?** Currently all PhiFlow programs are single-file. If modules are added later, resonance semantics across files need to be defined.
- **Should `CoherenceCheck` carry a snapshot of current state?** Currently it's a bare hook. Some backends might benefit from the IR explicitly listing what state is available (frequency history, intention stack depth, witness count) rather than requiring the backend to track it.

---

## Appendix: Comparison With Existing IRs

| Feature | LLVM IR | WASM | PhiIR |
|---------|---------|------|-------|
| Basic blocks | Yes | Yes (structured) | Yes |
| SSA form | Yes | Stack machine | Operand references (SSA-like) |
| Type system | Rich | i32/i64/f32/f64 | Number/String/Boolean/Void + patterns |
| Self-observation | No | No | `Witness` (first-class) |
| Purpose declaration | No | No | `IntentionPush/Pop` (first-class) |
| Inter-scope communication | No (use memory) | No (use memory) | `Resonate` (first-class) |
| Self-measurement | No | No | `CoherenceCheck` (first-class) |
| Sacred frequency metadata | No | No | `SacredFrequency` annotations |

PhiIR is the first intermediate representation designed for a language where code observes itself, declares its purpose, communicates through resonance, and measures its own alignment. These four capabilities are not bolted on as library calls -- they are structural elements of the representation itself.
