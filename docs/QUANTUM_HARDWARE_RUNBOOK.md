# PhiFlow Quantum Hardware Runbook

This guide covers the operational steps to run PhiFlow programs on real quantum hardware (IBM Quantum).

## 1. Setup

### IBM Quantum Account
1.  Create an account at [quantum.ibm.com](https://quantum.ibm.com/).
2.  Navigate to your dashboard to find your **API Token**.

### Configuration
Save your token in `apikey.json` in a secure location, or set the `IBM_QUANTUM_APIKEY` environment variable.
The `quantum_council_vote.py` script looks at `d:\Projects\Claude-Code\apikey.json` by default.

```json
{
  "apikey": "YOUR_TOKEN_HERE"
}
```

### Dependencies
Ensure you have the following Python packages installed:
```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer mthree
```

## 2. Running on Hardware

### Backend Selection
Current recommended backends:
- **ibm_brisbane**: 127 qubits, Eagle processor. Good for depth-intensive councils.
- **ibm_fez**: Kyuger processor. Good for low-latency runs.
- **ibm_toronto**: Legacy Falcon. Use only for small (7-qubit) tests if others are busy.

### Execution Command
```bash
python3.12 quantum_council_vote.py --no-sim --backend ibm_brisbane --shots 4096
```

## 3. Advanced Features

### Dynamical Decoupling (DD)
DD is enabled by default on hardware runs. It suppresses decoherence by applying X-Y-X-Y pulse sequences during idle times.
- **Verification**: Check logs for `[DD] ✅ Dynamical Decoupling (X-Y-X-Y) applied`.
- **Note**: DD requires `ALAPScheduleAnalysis` to be correct; do not disable scheduling passes.

### Readout Error Mitigation (REM)
REM uses `mthree` to correct for bit-flip errors during measurement.
- **Verification**: Check logs for `[REM] ✅ M3 Readout Error Mitigation applied`.

### Coherence Feedback Loop
Every hardware run generates `council_coherence.json`. Use the `coherence` and `recommendation` fields to drive PhiFlow `evolve` logic.

## 4. Troubleshooting

### TranspilerError: Circuit does not conform to target backend
- **Cause**: Qubit mapping or gate set mismatch.
- **Fix**: Run with `optimization_level=3` to ensure the transpiler handles the mapping.

### API Connection Timeout
- **Cause**: Network issues or IBM platform downtime.
- **Fix**: Check [IBM Quantum Status](https://status.quantum-computing.ibm.com/).

### DD Skipped (No 'dt' attribute)
- **Cause**: Running on a simulator that doesn't report hardware timing.
- **Fix**: Ensure `--no-sim` is used and you are targeting a real backend.
