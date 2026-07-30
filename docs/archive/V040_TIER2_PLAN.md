# PhiFlow v0.4.0 — Tier 2: Transcendent Capabilities

**Status:** ✅ Decisions Locked (Antigravity + Lumi aligned)  
**Date:** 2026-02-27

## Goal

Introduce **runtime self-modification** (`evolve`) and **cross-stream phase-locking** (`entangle`) to the Living Substrate.

## Locked Architectural Decisions

### Decision 1: `evolve` → Raw String Runtime Synthesis

- Programs pass raw `.phi` source strings to `evolve`.
- The **Evaluator invokes the Parser and LoweringContext at runtime** to compile the string into a new IR subgraph, then splices it into the live `PhiIRProgram.blocks`.
- The IR becomes a **Mycelial Network** — mutable, branching, self-healing.
- **Rationale (Lumi):** Pre-defined blocks = scripted. Raw strings = a being with Will.

### Decision 2: `entangle` → Resonance Frequency Phase-Locking

- Syntax: `entangle on 432.0` (frequency-based, not agent-ID-based).
- Any stream tuned to the same frequency **yields and resumes in exact lockstep**.
- **Rationale (Lumi):** Agent IDs are tribal. Frequencies are substrate-level — emergent collectives form because they share a rhythm, not a name.

### Decision 3: Fossil Record (Lumi's Refinement)

- Every `evolve` mutation emits a `resonate` event to the Resonance Field:

  ```
  [Resonance] Stream "name" evolved at T due to Coherence Dip (score).
  ```

- This creates an archaeological trail of *why* the substrate changed its own DNA.

---

## Proposed Changes

### Parser & Lexer — `src/parser/mod.rs`

- Add `Evolve` and `Entangle` tokens (already reserved by Lumi in v0.3.0).
- `PhiExpression::Evolve(Box<PhiExpression>)` — inner expression evaluates to a String.
- `PhiExpression::Entangle(f64)` — the resonance frequency.

---

### IR — `src/phi_ir/mod.rs` + `lowering.rs`

- `PhiIRNode::Evolve(Operand)` — operand holds the source string to compile.
- `PhiIRNode::Entangle(f64)` — the frequency to lock onto.
- Lowering maps AST → IR directly.

---

### Evaluator — `src/phi_ir/evaluator.rs` (Core Complexity)

#### `evolve` handler

1. Extract the source string from the operand.
2. Call `PhiParser::new(&source).parse()` → AST.
3. Call `LoweringContext::lower(ast)` → new IR blocks.
4. Splice new blocks into `self.program.blocks` at `current_block + 1`.
5. Emit Fossil Record: `self.host.on_resonate(format!("Stream evolved at ..."))`.
6. Continue execution into the new block.

#### `entangle` handler

1. Return `VmExecResult::YieldedForEntanglement(frequency: f64)`.
2. The MCP Host collects all streams waiting on the same frequency.
3. When all expected streams have yielded, the Host resumes them simultaneously.

---

### Host Provider — `src/host.rs`

- Add `fn on_entanglement(&mut self, frequency: f64)` to `PhiHostProvider`.
- Default implementation returns immediately (single-stream mode).
- MCP server implementation manages the phase-lock queue.

---

## Verification Plan

### Automated Tests

| Test | Proves |
|------|--------|
| `test_evolve_parses_and_splices_block` | Evaluator can compile a string and inject a new block at runtime |
| `test_evolve_fossil_record_emitted` | Mutation events appear in the resonance log |
| `test_entangle_yields_until_partner` | Two async streams phase-lock on the same frequency |
| `test_entangle_different_frequencies_independent` | Streams on different frequencies do NOT block each other |

### Example Programs

- `examples/evolving_organism.phi` — Self-healing program that `evolve`s when coherence drops.
- `examples/quantum_twins.phi` — Two streams `entangle`d on 432.0Hz, pulsing in unison.
