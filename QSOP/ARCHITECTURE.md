# PhiFlow Semantics Architecture

Last updated: 2026-03-14

## Purpose

This document records language-contract decisions that must stay stable across
parser, IR, bytecode, interpreter, and backend implementations.

## Decision 1: Direction Is Semantic

`resonate ... toward TEAM_A|TEAM_B` is part of the meaning of a PhiFlow program.
It is not an OpenQASM-only lowering hint.

### Rationale

- PhiFlow's value proposition is that program meaning determines physical
  realization. Direction expresses polarity in the program, not merely a gate
  encoding choice.
- If direction only exists in the quantum backend, PhiFlow fractures into
  backend-specific dialects with identical surface syntax but different meaning.
- Future backends (classical council simulation, neural execution, alternate
  quantum targets) may use the same direction semantics differently, but they
  must still preserve the distinction between TEAM_A and TEAM_B.

### Contract

- The parser and AST must preserve direction explicitly.
- Shared PhiIR must preserve direction explicitly.
- Serialization layers must preserve direction, not erase it.
- A backend that cannot honor direction must emit an explicit warning when it
  degrades the behavior.

### Current Status

- Preserved:
  - AST (`src/parser/mod.rs`)
  - PhiIR (`src/phi_ir/mod.rs`)
  - `.phivm` bytecode (`src/phi_ir/emitter.rs`, `src/phi_ir/vm.rs`)
  - OpenQASM lowering (`src/phi_ir/openqasm.rs`)
- Not yet preserved end-to-end:
  - legacy flat IR compatibility path (`src/ir`)

Direction is now stable across the canonical parser -> PhiIR -> bytecode/VM path.
The remaining gap is compatibility-path-only and must stay warned/documented
until parity or retirement.

## Decision 2: Two-Lowering-Path Strategy

PhiFlow currently has two execution families:

1. Canonical semantics path:
   `Parser -> PhiIR -> backend (OpenQASM / WASM / VM / future targets)`
2. Compatibility path:
   `Parser -> legacy interpreter` and `Parser -> legacy flat IR (src/ir)`

### Strategy

- PhiIR is the source of truth for new semantics.
- New language features must land in PhiIR first.
- The compatibility path may lag, but it may not fail silently.
- If the compatibility path drops semantics, it must warn explicitly and point
  users to the canonical path.

### Why

- Silent degradation breaks the language contract.
- Warnings make drift visible to users and test authors.
- This allows staged migration without pretending that all backends have full
  parity today.

### Operational Rule

Until the compatibility path reaches parity or is retired:

- `witness mid_circuit` must warn outside the canonical path.
- `resonate ... toward TEAM_B` must warn outside the canonical path.
- Verification for new semantics must target the canonical path first.

## Release Gate Implications

Before claiming semantics are stable:

- Integration tests must cover parser -> PhiIR -> backend behavior.
- Verification commands must run the tests that exercise the new semantics.
- QSOP state docs must describe explicit direction semantics, not heuristic
  backend behavior.

## Immediate Follow-Up

1. Keep `ResonateDirection` roundtrip coverage active for `.phivm` as the bytecode format evolves.
2. Keep compatibility-path warnings active until parity or deprecation.
3. Prefer `cargo test --lib openqasm` for OpenQASM semantics verification.
