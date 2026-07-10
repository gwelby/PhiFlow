#!/usr/bin/env python3.12
"""
Poll multiple GHZ scaling jobs and aggregate coherence results.
Usage:
  python3.12 scripts/poll_ghz_scaling.py <job_id_1:n_1> <job_id_2:n_2> ...
Example:
  python3.12 scripts/poll_ghz_scaling.py d98fsc0tcv6s73dm35k0:4 d98fsf52su3c739j82bg:5 ...
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from poll_ibm_real import poll_job, calculate_coherence


def main():
    if len(sys.argv) < 2:
        print("Usage: python3.12 poll_ghz_scaling.py [--no-wait] <job_id>:<n> [<job_id>:<n> ...]")
        sys.exit(1)

    wait = '--no-wait' not in sys.argv
    job_args = [arg for arg in sys.argv[1:] if arg != '--no-wait']

    results = []
    for arg in job_args:
        job_id, n = arg.rsplit(':', 1)
        n = int(n)
        print(f"\n{'='*60}")
        print(f"Polling n={n} job {job_id}")
        print(f"{'='*60}")
        counts = poll_job(job_id, wait=wait, max_wait_minutes=60)
        if counts is None:
            print(f"Failed to get counts for n={n}")
            continue
        coherence = calculate_coherence(counts)
        total = sum(counts.values())
        results.append({
            "n_qubits": n,
            "job_id": job_id,
            "shots": total,
            "coherence": coherence,
            "counts": counts,
        })
        print(f"  n={n}: coherence={coherence:.4f} ({total} shots)")

    results.sort(key=lambda x: x["n_qubits"])
    summary_path = "/tmp/ghz_scaling_results.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("GHZ SCALING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  n={r['n_qubits']:>2}: coherence={r['coherence']:.4f}  job={r['job_id']}")
    print(f"\nFull results saved to: {summary_path}")


if __name__ == '__main__':
    main()
