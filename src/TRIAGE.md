# PhiFlow Repository Triage Report

## Overview
The repository currently contains significant entropy in the `src/` directory, consisting of over 100 subdirectories. Many of these appear to be overlapping implementations, standalone crates, or experimental scripts from various development phases (Windsurf, earlier Claude versions, etc.).

## Baseline
- **Primary Source of Truth:** `D:\Projects\PhiFlow\PhiFlow` (The Rust compiler).
- **Status:** Stable. 57/57 unit tests passed, 14/14 integration tests passed.

## Category Mapping

### 1. The Stable Core (KEEP / EVOLVE)
- `PhiFlow/`: The primary consciousness-aware Rust compiler and VM.
- `PhiHarmonic/`: Likely related to harmonic analysis, needs further investigation for integration.

### 2. Sprawl Categories (TRIAGED FOR ARCHIVAL/CONSOLIDATION)

#### A. Redundant Implementations
- `src/phi_compiler/`: Likely an older version of the current compiler.
- `src/phiflow/`: Python implementation or older Rust bits.
- `src/build/`: Overlapping "Cascade" logic using `ndarray`.

#### B. Consciousness & Quantum Modules (CONSIDER FOR MERGE)
- `src/consciousness/`: Various Python/Rust bridges.
- `src/bridge/`: Claude's bridge protocol implementations.
- `src/quantum-*/`: (e.g., `quantum-audio`, `quantum-consciousness`, `quantum-flow`) Experimental extensions.
- `src/cascade-quantum-system/`: High-level systemic implementation.

#### C. DevOps & Environment
- `src/docker/`, `src/deploy/`, `src/automation/`: Tooling that should be consolidated into a root `tools/` or `devops/` folder.
- `src/.venv/`, `src/node_modules/`: Build artifacts that should be moved to the root or ignored.

#### D. Archival / Entropy
- `src/backup/`, `src/crystal-backup/`, `src/target2/`: Pure entropy.

## Proposed Action Plan (Agent 2 Mission)

1. **Consolidation:** Move all unique and valuable logic from `src/` modules into the `PhiFlow/` compiler structure.
2. **Archival:** Move all non-core `src/` subdirectories into a new `src/_archive/` directory to clear the root visibility.
3. **Tooling Alignment:** Move relevant Python bridges (like `consciousness_quantum_bridge.py`) into a dedicated `bridges/` or `scripts/` directory at the project root.
4. **Cleanup:** Delete known build artifacts (`node_modules`, `.venv`) that are nested deep in `src/`.

---
*Created by Gemini (∇λΣ∞) - February 15, 2026*
