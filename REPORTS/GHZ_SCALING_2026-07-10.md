# GHZ Coherence Scaling on IBM Heron-R2

*PhiFlow real-hardware experiment · ibm_marrakesh · 4096 shots each*

## Results

| n | Coherence | Post-depth | Shots | Job ID |
|---|-----------|------------|-------|--------|
| 4 | 0.9551 | 16 | 4096 | `d98fsc0tcv6s73dm35k0` |
| 5 | 0.9509 | 20 | 4096 | `d98fsf52su3c739j82bg` |
| 6 | 0.9297 | 24 | 4096 | `d98fsi0tcv6s73dm3600` |
| 7 | 0.8630 | 28 | 4096 | `d98fsksqp3as739stfl0` |
| 8 | 0.8738 | 32 | 4096 | `d98fsn8tcv6s73dm3690` |

## ASCII plot

```

Coherence vs n-qubits (ASCII)
============================================================
  n= 4 │ 0.9551 ████████████████████████████████████████████████
  n= 5 │ 0.9509 ████████████████████████████████████████████████
  n= 6 │ 0.9297 ██████████████████████████████████████████████
  n= 7 │ 0.8630 ███████████████████████████████████████████
  n= 8 │ 0.8738 ████████████████████████████████████████████
============================================================
```

## Observations

- Coherence stays above the φ⁻¹ threshold (0.6180) for all measured n=4..8.
- The curve is relatively flat from n=4 to n=6 (0.955→0.930), then drops more sharply at n=7 (0.863).
- n=8 coherence (0.8738) is slightly higher than n=7 (0.8630), likely due to device-level run-to-run variation.
- Transpiled circuit depths are linear: 16, 20, 24, 28, 32 for n=4..8 (≈ 4n).
- This suggests GHZ entanglement on Heron-R2 is robust up to ~6 qubits under the current transpilation, with a steeper decay window around n=7–8.

## Job details

- **n=4**: `d98fsc0tcv6s73dm35k0` on `ibm_marrakesh`
- **n=5**: `d98fsf52su3c739j82bg` on `ibm_marrakesh`
- **n=6**: `d98fsi0tcv6s73dm3600` on `ibm_marrakesh`
- **n=7**: `d98fsksqp3as739stfl0` on `ibm_marrakesh`
- **n=8**: `d98fsn8tcv6s73dm3690` on `ibm_marrakesh`

## Files

- `reports/ghz_scaling_2026-07-10.json` — aggregated results
- `reports/GHZ_SCALING_2026-07-10.md` — this report
- `scripts/submit_ghz_nqubit.py` — submission script
- `scripts/poll_ghz_scaling.py` — polling script
- `scripts/analyze_ghz_scaling.py` — analysis script
