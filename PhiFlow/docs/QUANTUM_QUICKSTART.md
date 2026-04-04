# Quantum Council Vote: Quickstart Guide

This guide will show you how to run your first quantum council vote using PhiFlow and the OpenQASM 3.0 backend in under 5 minutes.

## Prerequisites

-   **Rust Toolchain:** `cargo` and `rustc` installed.
-   **Python 3:** For running the quantum post-processor.
-   **Qiskit:** For simulating or running on real hardware.
    ```bash
    pip install qiskit qiskit-aer
    ```

## 1. Create Your Program

Create a file named `my_vote.phi` with the following content:

```phi
// A simple two-master council vote
intention "Master Tesla" {
    resonate 0.85 toward TEAM_A
    entangle on 432
}

intention "Master Einstein" {
    resonate 0.72 toward TEAM_B
    entangle on 432
}

// Observe the state (this collapses the quantum circuit)
witness
```

## 2. Compile to OpenQASM

Run the PhiFlow compiler to generate the OpenQASM 3.0 circuit:

```bash
cargo run --release --bin phic -- --target openqasm my_vote.phi > my_vote.qasm
```

Optional: Use `--optimize-depth` for hardware-optimized circuits.

## 3. Run the Post-Processor

Use the `quantum_council_vote.py` script to run the simulation and see the results:

```bash
python3 D:/Projects/Gambling/quantum/quantum_council_vote.py --simulate --game "My First Quantum Vote"
```

*Note: You may need to update the path to the script if you are in a different directory.*

## 4. Analyze the Results

The script will output a report similar to this:

```
============================================================
  QUANTUM COUNCIL VOTE - My First Quantum Vote
============================================================
  Matchup : TEAM_A vs TEAM_B
  Line    : 0
  Masters : 2 voting

  PICK    : TEAM_A
  Vote    : 56.5% toward TEAM_A
  Council Confidence: 18.2%

  Kelly (full):    12.4% of bankroll
  Kelly (1/4):     3.1% of bankroll  <- RECOMMENDED

  WITNESS: Tesla and Einstein share resonance field at 432Hz.
============================================================
```

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
