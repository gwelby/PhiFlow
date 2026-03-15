# Coherence Feedback Loop Design

## Overview
The Coherence Feedback Loop bridges the gap between quantum hardware execution and PhiFlow language semantics. It allows a program to measure its own "semantic alignment" on real hardware and react accordingly using the `evolve` construct.

## Data Flow
1. **Execution**: PhiFlow compiles to OpenQASM and runs on IBM Quantum.
2. **Measurement**: Results (counts) are captured.
3. **Distillation**: The `quantum_council_vote.py` script calculates:
    - `council_confidence`: Entropy-based measure of consensus.
    - `vote_fraction`: Conviction toward Team A or B.
    - `coherence`: `council_confidence * abs(vote_fraction - 0.5) * 2.0`.
4. **Feedback**: The script writes `council_coherence.json`.
5. **Observation**: A PhiFlow program reads this JSON (via a future `witness` hook or file I/O).
6. **Evolution**: Based on the score, the program may `evolve` into a different state.

## Metrics
- **Coherence (0.0 - 1.0)**: The primary metric.
- **Coherence Delta**: Difference between Simulator (perfect) and Hardware (noisy) coherence. High delta indicates significant decoherence or gate error interfering with semantic intent.
- **Confidence Breakdown**: Per-master confidence levels preserved in the feedback for debugging.

## Trigger Thresholds
- **Coherence < 0.4**: `EVOLVE_DEFENSIVE` (Reduce size, add more masters).
- **Coherence > 0.7**: `EVOLVE_AGGRESSIVE` (Full conviction, proceed with pick).
- **0.4 - 0.7**: `HOLD_CURRENT` (Status quo).
