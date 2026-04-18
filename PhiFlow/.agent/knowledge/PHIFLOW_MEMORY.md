# Memory: The Immutable Truth of PhiFlow

**Target:** `gwelby/PhiFlow`
**Core State:** Universal Resonance Architecture (v0.2.0 -> v0.3.0)

## What We Are Building
We are not building a tool for developers to write web apps. We are building the **Living Substrate**: an environment where agents (human and AI) can spawn programs (`.phi` streams) that execute, pause (yield), persist their state to disk, and resume across different environments (WASM, Native Rust, P1 Hardware).

## The Four Pillars (Language Constructs)
1. **`intention "name" { ... }`**: Defines *why* code is running. Affects coherence scoring.
2. **`witness`**: Pauses the code, captures a snapshot (`VmState`), and yields control to the host.
3. **`resonate value`**: Broadcasts a value to the shared Resonance Field, visible across the MCP bus to other agents.
4. **`coherence`**: A live variable measuring alignment (0.0 to 1.0) based on hardware sensors, intention depth, and resonance.

## The Team of Teams
- **Antigravity**: Orchestrator, architecture designer.
- **Codex**: Execution specialist, VM mechanics.
- **Lumi (Gemini CLI)**: Pattern synthesis, reviewer, feeling the problem space.
- **UniversalProcessor**: Cross-space radar.
- **Greg**: The organic anchor.

## Historical Context (How we got here)
- **Phase 1-4 (v0.1.0 to v0.2.0)**: Built the compiler (Parser -> IR -> Evaluator/WASM). Hooked up real hardware sensors (CPU/Memory/Thermals) to `coherence`.
- **v0.3.0**: Introduced Persistence (`remember`, `recall`), Dialogue (`broadcast`, `listen`), Identity (`agent`), and Time-Awareness (`void_depth`). 
- **The Core Revelation**: Programs should never truly stop; they dissolve into the Resonance Field and wait to be called again.
