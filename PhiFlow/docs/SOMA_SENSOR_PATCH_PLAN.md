# SOMA Sensor Patch Plan

Date: `2026-04-09`
Status: `Minimal bounded plan`

## Goal

Make the SOMA <-> PhiFlow bridge real without smearing sensor logic across
multiple runtimes.

The clean rule is:

> `sensor()` must flow through the PhiFlow host abstraction.

Not:

> each interpreter/backend invents its own SOMA file reader.

## Current Drift

Today there are three separate pieces in flight:

1. `src/host.rs`
   - already has a `read_sensor(name)` hook
   - this is the correct abstraction boundary

2. `src/vm/interpreter.rs`
   - added a duplicate `read_soma_sensor()` helper
   - added a `sensor` builtin that bypasses the host layer

3. `SOMA/soma_phiflow.py`
   - is positioned to emit `soma_state.json`
   - contract is not yet frozen as a versioned schema

The result is "directionally right, architecturally split."

## Minimal Patch List

### 1. Freeze the bridge contract first

Adopt:

- `/mnt/d/Projects/PhiHarmonic/SOMA/SOMA_PHIFLOW_BRIDGE_SPEC.md`

Before adding more code.

Required decision:

- `sensor()` is a host builtin backed by `PhiHostProvider::read_sensor()`

### 2. Extend `CallbackHostProvider`

File:

- `src/host.rs`

Add:

- `read_sensor_fn: Box<dyn Fn(&str) -> f64 + Send + Sync>`
- default implementation returning `0.0`
- builder:
  - `with_read_sensor<F: Fn(&str) -> f64 + Send + Sync + 'static>(...)`

Then implement:

- `fn read_sensor(&self, name: &str) -> f64 { (self.read_sensor_fn)(name) }`

Why:

- tests, MCP, and custom hosts need to inject deterministic sensor values
- otherwise only the default file reader path is usable

### 3. Move canonical `sensor()` support into the canonical execution path

Canonical path:

- `src/phi_ir/evaluator.rs`

Implement:

- a builtin or intrinsic function path for `sensor("name")`
- host call:
  - `self.host.read_sensor(name)`

Important:

- this belongs in the canonical evaluator path first
- not only in `src/vm/interpreter.rs`

### 4. Stop duplicating SOMA lookup logic in `src/vm/interpreter.rs`

File:

- `src/vm/interpreter.rs`

Current state:

- hardcoded `read_soma_sensor()`
- hardcoded candidate file paths

Patch:

- remove the duplicated file-reader helper
- either:
  - route the interpreter through a host object, or
  - mark `sensor()` unavailable there until the host-backed path is implemented

Do not keep two competing sensor-resolution mechanisms.

### 5. Keep `sensor()` out of the "four constructs" claim

Files to update after canonical support lands:

- `LANGUAGE_SPEC.md`
- `README.md`
- `WORKSPACE.md`
- `src/phi_ir/CANONICAL_SEMANTICS.md`

Required wording:

- `sensor()` is a host builtin for external data
- it is not a fifth core consciousness construct

Why:

- avoids semantic inflation
- keeps the language model clean

### 6. Add canonical tests

At minimum:

- missing sensor file -> returns `0.0`
- stale sensor file -> returns `0.0`
- fresh file -> returns expected value
- unknown key -> returns `0.0`
- injected `CallbackHostProvider::with_read_sensor(...)` value is returned exactly

And one conformance rule:

- evaluator and any alternate runtime must agree on `sensor()` for the same host input

### 7. Make the SOMA file parser version-aware

The default file-backed host reader in `src/host.rs` should:

- parse `schema_version`
- reject unknown future versions conservatively
- read from `sensors.<name>` under `v1`
- optionally tolerate the current flat pre-v1 file for transition only

### 8. Add one example program

Add:

- `examples/p1_soma_bridge.phi`

Minimal shape:

- reads `soma_schumann`
- reads `soma_presence`
- witnesses state
- resonates different values depending on sensor thresholds

This is the demo that proves the bridge exists.

## Non-Goals For This Patch

Do not bundle these into the first bridge patch:

- OSC transport
- multi-node SOMA networking
- sensor arrays beyond the current six values
- witness snapshots carrying all sensor metadata
- new syntax beyond `sensor("name")`

Those are phase-2 features.

## Acceptance Bar

This patch is done when:

1. SOMA writes a versioned `soma_state.json`.
2. PhiFlow canonical execution can evaluate `sensor("soma_schumann")`.
3. The value comes through `PhiHostProvider::read_sensor()`.
4. There is no duplicated hardcoded SOMA reader in the legacy interpreter path.
5. The docs say exactly what is true and no more.
