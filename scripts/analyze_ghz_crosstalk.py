#!/usr/bin/env python3.12
"""
Analyze GHZ-6 crosstalk test results.

Computes GHZ coherence from the first 6 measurement bits only, ignoring
spectator bits. Also reports spectator error (fraction of shots where any
spectator qubit flipped from |0⟩ to |1⟩).

Usage:
  python3.12 scripts/analyze_ghz_crosstalk.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from poll_ibm_real import poll_job


# Job IDs from submit_ghz_crosstalk.py
CROSSTALK_JOBS = {
    0: "d98scdcqp3as739tbe3g",
    2: "d98scdsqp3as739tbe40",
    4: "d98sce2f47jc73a8a8ag",
    5: "d98sceaf47jc73a8a8bg",
}


def load_counts(job_id, n_ghz=6):
    """Load counts from individual count file or poll job if needed."""
    counts_path = Path(f"/tmp/ibm_counts_{job_id}.json")
    if counts_path.exists():
        with open(counts_path) as f:
            counts = json.load(f)
        return counts

    print(f"Polling job {job_id}...")
    counts = poll_job(job_id, wait=True, max_wait_minutes=60)
    if counts is None:
        raise RuntimeError(f"Failed to poll job {job_id}")
    return counts


def compute_ghz_coherence(counts, n_ghz=6):
    """Compute coherence of the first n_ghz qubits, ignoring spectators."""
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0

    ghz_all_zeros = "0" * n_ghz
    ghz_all_ones = "1" * n_ghz

    ghz_good = 0
    for state, c in counts.items():
        ghz_state = state[:n_ghz]
        if ghz_state == ghz_all_zeros or ghz_state == ghz_all_ones:
            ghz_good += c

    return ghz_good / total, total


def compute_spectator_error(counts, n_ghz=6):
    """Compute fraction of shots where any spectator qubit flipped to |1⟩."""
    total = sum(counts.values())
    if total == 0:
        return 0.0

    spectator_bad = 0
    for state, c in counts.items():
        spectator_state = state[n_ghz:]
        if any(b == "1" for b in spectator_state):
            spectator_bad += c

    return spectator_bad / total


def main():
    results = []
    for k, job_id in sorted(CROSSTALK_JOBS.items()):
        counts = load_counts(job_id)
        coherence, total = compute_ghz_coherence(counts)
        spectator_error = compute_spectator_error(counts)

        results.append({
            "k_spectators": k,
            "job_id": job_id,
            "shots": total,
            "ghz_coherence": coherence,
            "spectator_error": spectator_error,
            "counts": counts,
        })

    print("\n" + "=" * 70)
    print("GHZ-6 CROSSTALK ANALYSIS — ibm_marrakesh")
    print("=" * 70)
    print(f"{'k':>3} | {'GHZ coherence':>14} | {'spectator error':>15} | {'shots':>6} | job_id")
    print("-" * 70)
    for r in results:
        print(
            f"{r['k_spectators']:>3} | {r['ghz_coherence']:>14.4f} | "
            f"{r['spectator_error']:>15.4f} | {r['shots']:>6} | {r['job_id']}"
        )
    print("=" * 70)

    # Save results
    out_path = "/tmp/ghz_crosstalk_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
