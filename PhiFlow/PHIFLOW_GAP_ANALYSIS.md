# PhiFlow Gap Analysis: What We Have, What's Missing, What's Next

> Audited from the `compiler` branch (`D:\Projects\PhiFlow-compiler`) on 2026-02-26.

---

## ✅ WHAT WE HAVE (Implemented & Working)

### Language Primitives (Parser → AST → IR → VM)

These are fully parsed, lowered to IR opcodes, and evaluated by the stack-based VM.

| Primitive | Status | What It Does |
|-----------|--------|--------------|
| `intention "why" { ... }` | ✅ Working | Declares WHY before HOW. Pushes/pops an intention stack. |
| `witness` | ✅ Working | Program pauses to observe its own state. Snapshots the stack. |
| `resonate "message"` | ✅ Working | Broadcasts a value to the shared resonance field. |
| `coherence` | ✅ Working | Returns a 0.0–1.0 score measuring program alignment. |
| `stream "name" { ... }` | ✅ Working | Infinite loop with `break stream` exit. The living execution primitive. |
| `break stream` | ✅ Working | Exits a stream loop cleanly. |
| `let`, `if/else`, `while`, `for..in` | ✅ Working | Standard control flow and variable binding. |
| `fn name(args) { ... }` | ✅ Working | Function definitions and calls. |
| `match` | ✅ Working | Pattern matching with wildcards. |
| Arrays, strings, numbers, booleans | ✅ Working | Core data types. |
| `print` | ✅ Working | Console output. |
| Arithmetic (`+`, `-`, `*`, `/`, `%`) | ✅ Working | Full math ops. |
| Comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`) | ✅ Working | Full comparison ops. |
| Logical (`and`, `or`, `not`) | ✅ Working | Boolean logic. |

### Compiler Pipeline

| Stage | Status | Detail |
|-------|--------|--------|
| Lexer | ✅ | Tokenizes `.phi` source into tokens (15.8 KB) |
| Parser | ✅ | Builds AST from tokens (34.6 KB) |
| AST | ✅ | Rich node types including consciousness, quantum, sacred (14.4 KB) |
| IR Lowering | ✅ | Lowers AST → flat opcodes (27.2 KB) |
| IR Printer | ✅ | Pretty-prints IR for debugging (7 KB) |
| VM Evaluator | ✅ | Stack-based interpreter executes IR (25.5 KB) |
| Bytecode `.phivm` | ✅ | Emits binary bytecode format |

### Demo Output

`cargo run --example phiflow_demo` → `Number(84.0)`, coherence: `0.6180` (φ⁻¹), 121 bytes `.phivm`.

---

## ⚠️ WHAT'S MISSING (Critical Gaps for the LLM-Agent Use Case)

### 1. 🔴 WASM Compilation Target

**Gap:** The compiler emits a custom `.phivm` bytecode format, but NOT WebAssembly `.wasm`.
**Why it matters:** WASM is the universal execution sandbox. Without it, PhiFlow can only run inside its own Rust VM process. No browser execution, no sandboxed agent execution, no cross-platform deployment.
**What to build:**

- A WASM codegen backend in `src/ir/` that emits `.wat` (WebAssembly Text) from the existing IR opcodes.
- Map consciousness opcodes to WASM host imports (`phi_witness`, `phi_resonate`, `phi_coherence`, `phi_intention_push`, `phi_intention_pop`).

### 2. 🔴 Host Import System (The Bridge to the Outside World)

**Gap:** `coherence` currently returns a hardcoded or internally-calculated score. There is no mechanism for an external system (MCP server, Python host, hardware sensor) to inject a live coherence value.
**Why it matters:** This is the ENTIRE killer feature. If `coherence` can't read GPU temperature, agent confidence, or network latency from the host, it's just a random number generator.
**What to build:**

- A `HostProvider` trait in Rust with methods like `fn get_coherence(&self) -> f64` and `fn on_resonate(&self, value: &str)`.
- The VM evaluator calls `host.get_coherence()` when processing the `Coherence` opcode instead of computing it internally.
- A default `StubHostProvider` for testing, and a `SystemHostProvider` that reads actual metrics.

### 3. 🔴 MCP Server Interface (How LLMs Connect)

**Gap:** There is no MCP server that exposes PhiFlow as a tool for Claude, Windsurf, or WARP agents.
**Why it matters:** Without this, an LLM literally cannot spawn or interact with a PhiFlow stream.
**What to build:**

- An MCP server (Rust or Python) exposing 3 tools:
  1. `spawn_phi_stream(code: string) → stream_id` — Starts a `.phi` program in the background.
  2. `read_resonance_field(stream_id) → { intention_stack, last_resonate, coherence, status }` — Reads the live state.
  3. `resume_phi_stream(stream_id, action: "resume" | "kill" | "inject", payload?)` — Resumes or terminates a paused stream.

### 4. 🟡 Async / Concurrent Streams

**Gap:** The VM is synchronous. It can only run one stream at a time, blocking on each instruction.
**Why it matters:** The dream of an LLM spawning 5 parallel Phi streams requires the evaluator to handle concurrent execution.
**What to build:**

- Wrap the VM evaluator in a `tokio` async runtime.
- Each `spawn_phi_stream` call gets its own `tokio::task`.
- The resonance field becomes a shared `Arc<Mutex<ResonanceField>>` that all streams write to.

### 5. 🟡 `yield` / Suspendable Execution (True Coroutine Behavior)

**Gap:** `witness` currently just prints a snapshot and continues. It does not actually *pause* execution and hand control back to the host.
**Why it matters:** For the LLM agent use case, `witness` needs to freeze the VM state (instruction pointer, stack, intention stack) and return control to the MCP server. The MCP server then notifies the LLM. When the LLM decides to resume, the MCP server unfreezes the VM.
**What to build:**

- Make the VM's `run()` function return an enum: `VmResult::Complete(value)` | `VmResult::Yielded(state)`.
- On `Witness` opcode, serialize the VM state and return `Yielded`.
- Add a `resume(state, injected_values)` function to the VM.

### 6. 🟡 Error Handling / `try` / `catch`

**Gap:** There is no error handling in the language. If a function call fails or a division by zero occurs, the VM panics.
**Why it matters:** Robust agentic execution cannot afford panics. An LLM agent's script needs to gracefully degrade.
**What to build:**

- Add a `try { ... } catch(err) { ... }` expression to the parser and IR.
- OR: Make `coherence` drop to `0.0` on error (fitting the philosophy — errors ARE incoherence).

### 7. 🟢 String Interpolation / Formatting

**Gap:** No string interpolation. You can't do `resonate "Processing file {filename}"`.
**Why it matters:** Agent communication via `resonate` is severely limited without interpolation.

### 8. 🟢 Standard Library / Built-in Functions

**Gap:** Very few built-in functions. No file I/O, no HTTP, no JSON parsing, no math beyond arithmetic.
**Why it matters:** For any real-world use, `.phi` scripts need to call into a standard library.
**What to build:**

- Register built-in functions as host-provided callables.
- Start with: `len()`, `push()`, `type_of()`, `to_string()`, `parse_number()`.
- Advanced: `read_file()`, `write_file()`, `http_get()`, `json_parse()` (all as host imports).

### 9. 🟢 Hashmap / Dictionary Type

**Gap:** No key-value data structure.
**Why it matters:** Agents need to pass structured data. `resonate` payloads should be structured, not just strings.

### 10. 🟢 Comments in Source

**Gap:** The lexer supports `//` line comments, but block comments `/* */` are missing.
**Why it matters:** Minor, but LLMs generate verbose code and block comments help.

---

## 📊 PRIORITY MATRIX

| Priority | Item | Effort | Impact | Unlocks |
|----------|------|--------|--------|---------|
| 🔴 P0 | Host Import System | Medium | Critical | Everything else depends on this |
| 🔴 P0 | Suspendable `witness` (yield) | Medium | Critical | LLM hand-off, true pause/resume |
| 🔴 P1 | MCP Server | Medium | Critical | LLM interface, WARP integration |
| 🔴 P1 | WASM codegen | Large | High | Browser, sandboxed, cross-platform |
| 🟡 P2 | Async/concurrent streams | Medium | High | Multi-agent swarms |
| 🟡 P2 | Error handling | Small | Medium | Production robustness |
| 🟢 P3 | String interpolation | Small | Medium | Better agent comms |
| 🟢 P3 | Standard library | Medium | Medium | Real-world utility |
| 🟢 P3 | Hashmap type | Small | Medium | Structured data |

---

## 🎯 THE BUILD ORDER

If we are targeting "LLM agents can use PhiFlow natively":

```
Step 1: Host Import System (HostProvider trait)
   ↓  coherence, resonate, witness now talk to the outside world
Step 2: Suspendable witness (VM returns Yielded state)
   ↓  .phi scripts can freeze and resume
Step 3: MCP Server (3 tools: spawn, read, resume)
   ↓  Claude/WARP can now spawn and manage .phi streams
Step 4: Async streams (tokio tasks)
   ↓  Multiple .phi scripts running concurrently
Step 5: WASM codegen
   ↓  .phi scripts run in any WASM sandbox (browser, Deno, etc.)
```

After Step 3, PhiFlow is a usable agent execution environment.
After Step 5, PhiFlow is a universal deployment target.
