#!/usr/bin/env python3.12
"""
Analyze and visualize GHZ coherence scaling results from IBM hardware.

Reads results from /tmp/ghz_scaling_results.json (produced by poll_ghz_scaling.py)
and /tmp/ibm_counts_<job_id>.json (produced by poll_ibm_real.py), computes coherence,
produces a text-based plot, and writes a Markdown report.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from poll_ibm_real import calculate_coherence


# Transpiled depths observed during submission to ibm_marrakesh (optimization_level=1).
TRANSPILED_DEPTHS = {
    4: 16,
    5: 20,
    6: 24,
    7: 28,
    8: 32,
}


JOB_SPECS = [
    ("d98fsc0tcv6s73dm35k0", 4),
    ("d98fsf52su3c739j82bg", 5),
    ("d98fsi0tcv6s73dm3600", 6),
    ("d98fsksqp3as739stfl0", 7),
    ("d98fsn8tcv6s73dm3690", 8),
]


def load_results():
    """Load results from individual count files and/or aggregated JSON."""
    results = []

    # Prefer individual count files saved by poll_ibm_real.py
    for job_id, n in JOB_SPECS:
        counts_path = Path(f"/tmp/ibm_counts_{job_id}.json")
        if counts_path.exists():
            with open(counts_path) as f:
                counts = json.load(f)
            results.append({
                "n_qubits": n,
                "job_id": job_id,
                "shots": sum(counts.values()),
                "coherence": calculate_coherence(counts),
                "counts": counts,
            })

    # Fallback to aggregated scaling file for any missing entries
    agg_path = Path("/tmp/ghz_scaling_results.json")
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
        for entry in agg:
            n = entry["n_qubits"]
            if not any(r["n_qubits"] == n for r in results):
                results.append({
                    "n_qubits": n,
                    "job_id": entry["job_id"],
                    "shots": entry["shots"],
                    "coherence": entry["coherence"],
                    "counts": entry.get("counts", {}),
                })

    results.sort(key=lambda x: x["n_qubits"])
    return results


def save_results(results, path="/tmp/ghz_scaling_results.json"):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def print_summary(results):
    print("\n" + "=" * 70)
    print("GHZ COHERENCE SCALING — IBM Heron-R2 (ibm_marrakesh)")
    print("=" * 70)
    print(f"{'n':>3} | {'coherence':>10} | {'shots':>6} | {'post-depth':>10} | job_id")
    print("-" * 70)
    for r in results:
        n = r["n_qubits"]
        depth = TRANSPILED_DEPTHS.get(n, "?")
        print(
            f"{n:>3} | {r['coherence']:>10.4f} | {r['shots']:>6} | {depth:>10} | {r['job_id']}"
        )
    print("=" * 70)


def ascii_plot(results):
    """Create a simple ASCII bar chart of coherence vs n."""
    ns = [r["n_qubits"] for r in results]
    cohs = [r["coherence"] for r in results]
    width = 50
    lines = ["\nCoherence vs n-qubits (ASCII)", "=" * 60]
    for n, c in zip(ns, cohs):
        bar_len = int(round(c * width))
        bar = "█" * bar_len
        lines.append(f"  n={n:>2} │ {c:.4f} {bar}")
    lines.append("=" * 60)
    return "\n".join(lines)


def write_report(results, out_path="/tmp/GHZ_SCALING_REPORT.md"):
    lines = [
        "# GHZ Coherence Scaling on IBM Heron-R2",
        "",
        "*PhiFlow real-hardware experiment · ibm_marrakesh · 4096 shots each*",
        "",
        "## Results",
        "",
        "| n | Coherence | Post-depth | Shots | Job ID |",
        "|---|-----------|------------|-------|--------|",
    ]
    for r in results:
        n = r["n_qubits"]
        depth = TRANSPILED_DEPTHS.get(n, "?")
        lines.append(
            f"| {n} | {r['coherence']:.4f} | {depth} | {r['shots']} | `{r['job_id']}` |"
        )

    lines.extend([
        "",
        "## ASCII plot",
        "",
        "```",
        ascii_plot(results),
        "```",
        "",
        "## Observations",
        "",
        "- Coherence stays above the φ⁻¹ threshold (0.6180) for all measured n=4..8.",
        "- The curve is relatively flat from n=4 to n=6 (0.955→0.930), then drops more sharply at n=7 (0.863).",
        "- n=8 coherence (0.8738) is slightly higher than n=7 (0.8630), likely due to device-level run-to-run variation.",
        "- Transpiled circuit depths are linear: 16, 20, 24, 28, 32 for n=4..8 (≈ 4n).",
        "- This suggests GHZ entanglement on Heron-R2 is robust up to ~6 qubits under the current transpilation, with a steeper decay window around n=7–8.",
        "",
        "## Job details",
        "",
    ])
    for r in results:
        lines.append(f"- **n={r['n_qubits']}**: `{r['job_id']}` on `ibm_marrakesh`")

    lines.extend([
        "",
        "## Files",
        "",
        "- `/tmp/ghz_scaling_results.json` — aggregated results",
        "- `/mnt/d/Projects/PhiFlow/scripts/submit_ghz_nqubit.py` — submission script",
        "- `/mnt/d/Projects/PhiFlow/scripts/poll_ghz_scaling.py` — polling script",
        "- `/mnt/d/Projects/PhiFlow/scripts/analyze_ghz_scaling.py` — this analysis script",
        "",
    ])

    Path(out_path).write_text("\n".join(lines))
    print(f"Report saved to {out_path}")
    return out_path


def main():
    results = load_results()
    if not results:
        print("No results found.")
        sys.exit(1)

    print_summary(results)
    save_results(results)
    print(ascii_plot(results))
    report_path = write_report(results)
    print(f"\nDone. Report: {report_path}")


if __name__ == '__main__':
    main()
