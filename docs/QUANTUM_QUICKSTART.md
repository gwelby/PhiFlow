# Quantum Council Vote: Quickstart Guide

This guide will show you how to run your first quantum council vote using PhiFlow and the OpenQASM 3.0 backend in under 5 minutes.

## Prerequisites

-   **Rust Toolchain:** `cargo` and `rustc` installed.
-   **Python 3:** For running the quantum post-processor.
-   **Qiskit:** For simulating or running on real hardware.
    ```bash
    pip install qiskit qiskit-aer
    ```

## 1. Locate the Example Program

The repository contains an example of a 3-qubit entangled council in `examples/quantum_council.phi`.

```phi
// quantum_council.phi
intention "observe" {
    witness
    resonate 0.618
    entangle on 432
}

intention "integrate" {
    witness
    resonate 0.618
    entangle on 432
}

intention "transcend" {
    witness
    resonate 0.618
    entangle on 432
    witness
}
```

## 2. Run the PhiFlow Compiler

Run the PhiFlow compiler targeting the quantum execution backend. This computes live coherence values per intention, generates parameterized OpenQASM 3.0, and executes the transpile guardrail before simulating or running on IBM hardware:

```bash
cargo run --release --bin phic -- --target quantum examples/quantum_council.phi
```

## 3. Analyze the Results

The CLI will output the captured coherence values for each intention, and display the generated OpenQASM 3.0 circuit substituting those values.

```
Compiling to PhiFlow IR...
🌌 Quantum Consciousness Council — parameterized emission
📊 Captured council coherence:
  observe: 0.3820
  transcend: 0.3520
  integrate: 0.3720
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

// Block entry
// Intention: observe
    ry(0.38196601125 * pi) q[0];
// Intention: integrate
    ry(0.37196601125 * pi) q[1];
    cx q[0], q[1]; // Entangle via 432Hz
// Intention: transcend
    ry(0.35196601125 * pi) q[2];
    cx q[1], q[2]; // Entangle via 432Hz

    // --- Final Witness measurements (end-of-circuit) ---
    c[0] = measure q[0]; // Final Witness q0
    c[1] = measure q[1]; // Final Witness q1
    c[2] = measure q[2]; // Final Witness q2
```

Note: IBM hardware integration is active. Ensure you have the `IBM_QUANTUM_TOKEN` in your Cascade vault if running live hardware jobs.

## Summary

You have successfully:
1.  Declared a **Semantic Intention**.
2.  Generated a **Physical Quantum Circuit** from that intention.
3.  Simulated a **Quantum Measurement** of the collective field.
4.  Analyzed the **Coherence** of the result.

**Next Steps:**
-   Add more masters to your council.
-   Use different sacred frequencies (528Hz, 594Hz) to create separate entanglement channels.
-   Check the `calibration_log.jsonl` file to see the historical performance of your runs.
