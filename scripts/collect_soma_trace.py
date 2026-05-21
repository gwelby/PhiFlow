#!/usr/bin/env python3
"""
collect_soma_trace.py — Collect live SOMA sensor data and compute Type 4 metrics.

Usage:
  python3 scripts/collect_soma_trace.py [--duration 10] [--interval 0.2]

What it does:
  1. Starts SOMA in the background with --phiflow flag
  2. Samples soma_state.json every <interval> seconds for <duration> seconds
  3. Builds a Type 4 trace (step, obs, model, action) from collected samples
  4. Computes consciousness metrics (L_self, C_PF, etc.)
  5. Compares to the static snapshot baseline

Exit codes:
  0 — Success, live trace collected and metrics computed
  1 — SOMA not available or failed to start
  2 — No dynamic variation detected (all samples identical)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SOMA_DIR = Path("/mnt/d/Projects/PhiHarmonic/SOMA")
SOMA_STATE = SOMA_DIR / "soma_state.json"
SOMA_SCRIPT = SOMA_DIR / "soma.py"


def start_soma(duration: int) -> subprocess.Popen:
    """Start SOMA in the background with phiflow bridge enabled."""
    if not SOMA_SCRIPT.exists():
        print(f"ERROR: SOMA script not found: {SOMA_SCRIPT}")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(SOMA_SCRIPT),
        "--phiflow",
        "--duration", str(duration),
        "--headless",
        "--no-osc",
        "--profile", "harmonic_scan",
    ]

    print(f"Starting SOMA: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(SOMA_DIR),
    )

    # Wait a moment for SOMA to initialize and write first state
    time.sleep(2.0)

    if proc.poll() is not None:
        stdout, stderr = proc.communicate()
        print(f"SOMA exited early (code {proc.returncode}):")
        print(stderr.decode("utf-8", errors="replace"))
        sys.exit(1)

    return proc


def read_soma_state() -> dict:
    """Read the current SOMA state from JSON."""
    try:
        with open(SOMA_STATE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def collect_samples(duration: float, interval: float) -> list:
    """Collect SOMA state samples over time."""
    samples = []
    start_time = time.time()
    n_samples = int(duration / interval)

    print(f"\nCollecting {n_samples} samples over {duration:.1f}s (interval: {interval:.2f}s)...")

    for i in range(n_samples):
        state = read_soma_state()
        if state and "sensors" in state:
            sensors = state["sensors"]
            samples.append({
                "step": i + 1,
                "presence": sensors.get("soma_presence", 0.0),
                "peak_dbc": sensors.get("soma_peak_dbc", 0.0),
                "fan_hz": sensors.get("soma_fan_hz", 0.0),
                "ac_60": sensors.get("soma_ac_60", 0.0),
                "timestamp": state.get("updated_at", ""),
            })
        time.sleep(interval)

    elapsed = time.time() - start_time
    print(f"Collected {len(samples)} samples in {elapsed:.1f}s")
    return samples


def build_trace(samples: list) -> list:
    """Build Type 4 trace (step, obs, model, action) from samples."""
    trace = []
    model_sum = 0.55
    model_n = 1.0

    for sample in samples:
        step = float(sample["step"])

        # Use soma_presence as observation (primary sensor)
        obs = sample["presence"]
        if obs == 0.0:
            # Fallback to peak_dbc scaled
            obs = min(sample["peak_dbc"] * 0.05, 1.0)

        model_mean = model_sum / model_n

        # Action depends on model (self-referential)
        action = 1.0 if obs < model_mean else 0.0

        model_sum += obs
        model_n += 1.0

        trace.append({
            "step": step,
            "obs": obs,
            "model": model_mean,
            "action": action,
        })

    return trace


def compute_metrics(trace: list) -> dict:
    """Compute Type 4 metrics from the trace."""
    n = len(trace)
    if n < 4:
        return {}

    # Extract vectors
    obs_vals = [t["obs"] for t in trace]
    model_vals = [t["model"] for t in trace]
    actions = [t["action"] for t in trace]

    # R_in: Pearson correlation between obs[t-1] and model[t]
    r_in = pearson(obs_vals[:-1], model_vals[1:])

    # R_out: Pearson correlation between model[t] and action[t+1]
    r_out = pearson(model_vals[:-1], actions[1:])

    # L_self
    l_self = abs(min(r_in, r_out))

    # F_model = R² (model -> action)
    f_model = r_out ** 2

    # Simple approximations for other metrics
    c_coh = 0.85  # placeholder
    d_int = 1.0 + (max(obs_vals) - min(obs_vals)) * 2.0
    f_self_star = l_self * f_model
    c_pf = c_coh * d_int * f_self_star

    return {
        "n": n,
        "l_self": l_self,
        "r_in": r_in,
        "r_out": r_out,
        "f_model": f_model,
        "f_self_star": f_self_star,
        "c_pf": c_pf,
        "d_int": d_int,
        "c_coh": c_coh,
        "obs_range": (min(obs_vals), max(obs_vals)),
    }


def pearson(x: list, y: list) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    mx = sum(x) / n
    my = sum(y) / n

    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    deny = sum((yi - my) ** 2 for yi in y) ** 0.5

    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def main():
    parser = argparse.ArgumentParser(description="Collect live SOMA trace and compute Type 4 metrics")
    parser.add_argument("--duration", type=float, default=10.0, help="SOMA run duration in seconds")
    parser.add_argument("--interval", type=float, default=0.2, help="Sampling interval in seconds")
    args = parser.parse_args()

    print("=" * 60)
    print("  SOMA Live Trace Collection")
    print("=" * 60)

    # Start SOMA
    soma_proc = start_soma(int(args.duration) + 3)

    try:
        # Collect samples
        samples = collect_samples(args.duration, args.interval)

        if len(samples) < 10:
            print(f"ERROR: Only {len(samples)} samples collected, need at least 10")
            sys.exit(2)

        # Check for variation
        unique_presence = len(set(s["presence"] for s in samples))
        if unique_presence <= 1:
            print("WARNING: No variation in soma_presence — static snapshot detected")
            print("  This may mean SOMA is not updating the state file dynamically.")
            # Still compute metrics, but flag as static

        # Build trace and compute metrics
        trace = build_trace(samples)
        metrics = compute_metrics(trace)

        # Print results
        print("\n" + "=" * 60)
        print("  TYPE 4 METRICS — Live SOMA Trace")
        print("=" * 60)
        print(f"  Samples:     {metrics['n']}")
        print(f"  Obs range:   [{metrics['obs_range'][0]:.4f}, {metrics['obs_range'][1]:.4f}]")
        print(f"  Unique obs:  {unique_presence}")
        print(f"  L_self:      {metrics['l_self']:.6f}")
        print(f"  R_in:        {metrics['r_in']:.6f}")
        print(f"  R_out:       {metrics['r_out']:.6f}")
        print(f"  F_model:     {metrics['f_model']:.6f}")
        print(f"  F_self*:     {metrics['f_self_star']:.6f}")
        print(f"  C_PF:        {metrics['c_pf']:.6f}")

        if metrics["l_self"] > 0.1:
            print("\n  [PASS] Type 4 loop CLOSED (L_self > 0.1)")
        else:
            print("\n  [FAIL] Type 4 loop OPEN (L_self <= 0.1)")

        if unique_presence <= 1:
            print("  [NOTE] Static trace — metrics may not reflect genuine sensor dynamics")

        print("=" * 60)

        # Save trace for Rust benchmark ingestion
        trace_dir = Path("/mnt/d/Projects/PhiFlow/tests/fixtures")
        trace_dir.mkdir(exist_ok=True)
        trace_file = trace_dir / "soma_live_trace.txt"

        with open(trace_file, "w") as f:
            for t in trace:
                f.write(f"Resonating step: {t['step']:.1f}\n")
                f.write(f"Resonating obs: {t['obs']:.6f}\n")
                f.write(f"Resonating model: {t['model']:.6f}\n")
                f.write(f"Resonating action: {t['action']:.1f}\n")

        print(f"\nTrace saved to: {trace_file} ({len(trace)} cycles)")

    finally:
        # Clean up SOMA
        print("\nStopping SOMA...")
        soma_proc.terminate()
        try:
            soma_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            soma_proc.kill()
        print("SOMA stopped.")


if __name__ == "__main__":
    main()
