# PhiFlow Engineering Specs: The Optimization Engine 🛠️

> **The Code:** Concrete implementation details for the optimization pipeline.

## 1. Data Structures

### 1.1 `OptimizationLevel` Enum

Located in `src/phi_ir/optimizer.rs`.

```rust
pub enum OptimizationLevel {
    Linear,          // Baseline: Standard recursive descent
    PhiEnhanced,     // Level 1: Constant folding, basic DCE
    PhiSquared,      // Level 2: Loop unrolling (factor ~1.618)
    PhiCubed,        // Level 3: Function inlining (depth 3)
    Quantum,         // Level 4: Parallelize independent blocks for QPU
    Consciousness,   // Level 5: Select algorithms based on "Intent"
}
```

### 1.2 `CoherenceState` Struct

Located in `src/consciousness/monitor.rs`.

```rust
pub struct CoherenceState {
    pub cpu_jitter: f64,          // 0.0 - 1.0 (inverse of std dev)
    pub quantum_fidelity: f64,    // 0.0 - 1.0 (from IBM Q provider)
    pub bio_feedback: f64,        // 0.0 - 1.0 (simulated or real EEG)
    pub total_coherence: f64,     // Geometric mean of above
}
```

## 2. Algorithms

### 2.1 The Phi-Harmonic Loop Unroller

* **Goal:** Optimize loop execution by unrolling based on Fibonacci sequence.
* **Trigger:** `OptimizationLevel::PhiSquared` or higher.
* **Logic:**
    1. Identify `While` or `For` loops in PhiIR.
    2. Check iteration count `N`.
    3. If `N` is constant, unroll completely (standard).
    4. If `N` is variable, unroll by factor `F` where `F` is the nearest Fibonacci number <= static capability.

### 2.2 Coherence Stabilization

* **Goal:** Prevent system crash/decoherence.
* **Trigger:** `CoherenceState.total_coherence < 0.95`.
* **Action:**
    1. **Throttle:** Insert `Sleep` nodes (measured in 432ms increments) to reduce CPU jitter.
    2. **Verify:** Re-run `Witness` node after sleep.
    3. **Fallback:** If coherence < 0.80, downgrade `OptimizationLevel` to `Linear`.

### 2.3 Quantum Block Identification

* **Goal:** Find code suitable for QPU offload.
* **Trigger:** `OptimizationLevel::Quantum`.
* **Logic:**
    1. Scan for `PhiIRBlock`s with no side effects (pure functions).
    2. Check for vectorizable operations (Matrix math, FFT).
    3. Tag block as `Candidate::Quantum`.
    4. *(Future)* Lower to QASM.

## 3. Interfaces

### 3.1 `Optimizer::optimize`

```rust
pub fn optimize(program: &mut PhiIRProgram, level: OptimizationLevel) -> Result<(), OptimizationError> {
    // 1. Run mandatory passes (DCE, Fold)
    // 2. Run level-specific passes
    // 3. Check coherence
    // 4. Return result
}
```

### 3.2 `CoherenceMonitor::check`

```rust
pub fn check() -> CoherenceState {
    // Aggregate metrics from hardware sensors/simulators
}
```

## 4. Acceptance Criteria

* [ ] `OptimizationLevel` enum implemented.
* [ ] Loop unrolling logic handles variable iteration counts.
* [ ] Coherence monitor returns valid 0.0-1.0 floats.
* [ ] Optimization downgrades automatically on low coherence simulation.
