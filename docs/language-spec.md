# PhiFlow Language Specification

**Version:** 0.4.0
**Status:** Implementation-accurate (reflects what the compiler actually does, not aspirational behavior)

## 1. Overview

PhiFlow is a statically-typed, interpreted language with five first-class primitives for runtime self-observation. Programs compile to three targets: a tree-walking evaluator, a bytecode VM, and WebAssembly.

## 2. The Five Primitives

### 2.1 `intention` — Purpose Scope

**Syntax:**
```
intention "name" {
    <statements>
}
```

**Semantics:** Pushes a named scope onto the intention stack. The stack depth is the primary input to the coherence formula. When the block exits, the scope is popped.

**Type:** The intention name is a string literal. The body is a block of statements.

**Runtime cost:** O(1) push/pop. The coherence value is recomputed on each `coherence` read, not on each push.

### 2.2 `witness` — Observation Point

**Syntax:**
```
witness                      // observe all current state
witness <expression>         // observe a specific value
```

**Semantics:** Pauses execution and calls the host's `witness` function with a snapshot of the current runtime state (intention stack, resonance field, coherence score, local variables, and optionally a specific expression value).

**Runtime cost:** O(n) where n is the size of the current state. The host decides what to do with the observation.

**Return value:** `void` — witness is a statement, not an expression.

### 2.3 `coherence` — Alignment Read

**Syntax:**
```
let c = coherence
```

**Semantics:** Reads the current coherence score C(d, k) as an f64 in [0, 1]. This is a built-in read-only value, not a function call. The runtime tracks the intention stack and resonance field continuously; `coherence` just reads the current value.

**Type:** `f64`

**Runtime cost:** O(1) — the value is computed from the current stack depth and resonance cardinality.

### 2.4 `resonate` — Inter-Scope Communication

**Syntax:**
```
resonate                     // share current scope's state to the field
resonate <expression>        // share a specific value to the field
resonate <expr> toward TEAM_B  // directional (for quantum lowering)
```

**Semantics:** Appends a value to `resonance_field[current_scope]`. The cardinality k of this vector is the secondary input to the coherence formula. Other scopes can read the resonance field.

**Type:** The expression can be any PhiFlow value (f64, string, bool, void).

**Runtime cost:** O(1) append.

### 2.5 `stream` — Self-Defining Loop

**Syntax:**
```
stream "name" {
    <statements>
    break stream             // exit the loop
}
```

**Semantics:** Pushes a stream scope, executes the body, and loops back to the body start on each iteration. `break stream` exits the loop. The stream has a name for observation and control.

**Runtime cost:** Same as a while loop. The stream scope is pushed/popped on each iteration.

## 3. Standard Language Constructs

### 3.1 Variables

```
let x = 42.0                 // untyped
let x: Number = 42.0         // typed
x = 43.0                     // assignment
```

### 3.2 Functions

```
function add(a: Number, b: Number) -> Number {
    return a + b
}
```

### 3.3 Control Flow

```
if <condition> { ... } else { ... }
while <condition> { ... }
for <var> in <iterable> { ... }
```

### 3.4 Operators

| Precedence | Operators |
|-----------|-----------|
| 1 (highest) | `^` (power) |
| 2 | unary `-`, `!` |
| 3 | `*`, `/`, `%` |
| 4 | `+`, `-` |
| 5 | `<`, `<=`, `>`, `>=` |
| 6 | `==`, `!=` |
| 7 | `&&` |
| 8 (lowest) | `\|\|` |

### 3.5 Types

- `Number` (f64)
- `String`
- `Bool`
- `Qubit`
- `Circuit`
- `Void`

### 3.6 Persistence (v0.3.0)

```
remember "key" = <expression>
recall "key"
broadcast "key" = <expression>
listen "key"
```

### 3.7 Agent Identity (v0.3.0)

```
agent "name" version "1.0.0" {
    <statements>
}
```

### 3.8 Integrity Anchoring (v0.5.0)

```
anchor "target" min_presence 0.8 frequency 432 {
    <statements>
}
```

## 4. Compilation Targets

### 4.1 Evaluator (Tree-Walking)

The default target. Parses to AST, evaluates directly. Used for development and testing.

### 4.2 Bytecode VM

Compiles AST to PhiIR, then to bytecode (`PHIV` format). The VM loads and executes the bytecode. Used for production execution.

**Binary format:** Magic `PHIV`, version 1, string table, block table, instruction stream.

**Opcodes:** 30+ opcodes covering all language constructs. Consciousness primitives are first-class opcodes:
- `OP_WITNESS` (0x30)
- `OP_INTENTION_PUSH` (0x31)
- `OP_INTENTION_POP` (0x32)
- `OP_RESONATE` (0x33)
- `OP_COHERENCE_CHECK` (0x34)
- `OP_STREAM_PUSH` (0x3C)
- `OP_STREAM_POP` (0x3D)

### 4.3 WebAssembly

Compiles PhiIR to WebAssembly Text format (.wat). The WASM module is pure — all consciousness semantics are host imports. See `docs/architecture.md` for the import mapping.

## 5. Error Handling

Parse errors produce `PhiDiagnostic` objects with:
- Error code (E001-E005)
- Line and column
- Found token
- Expected token (when applicable)
- Hint message
- Example fix

Runtime errors produce `VmError` with descriptive messages. The VM has a step limit to prevent infinite loops.
