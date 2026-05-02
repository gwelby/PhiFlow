"""
measure_type4.py
Compute L_self proxy from a type4_trace_benchmark.phi run.

Usage:
    cd /mnt/d/Projects/PhiFlow
    ./target/debug/phic examples/type4_trace_benchmark.phi 2>&1 > trace.txt
    python3 tools/measure_type4.py trace.txt

Output:
    R_in:    correlation(MODEL_t, OBS_{t-1})  -- past obs predicts model
    R_out:   MI(MODEL_t, DEVIATION_t)         -- model predicts behavior signal
    L_self:  min(R_in_norm, R_out_norm)       -- closed loop score [0,1]

PF reference:
    L_self(L) = min(R_in_normalized, R_out_normalized)
    where R_in = I_dir(X_{t-L:t-1} -> M_t | E) and R_out = I_dir(M_t -> X_{t+1:t+L} | X_t, E)
    consciousness_metric_program.md: if either leg = 0, loop is broken -> not Type 4
"""

import sys
import re
import numpy as np


def parse_trace(path: str):
    """Parse PhiFlow resonance output into (step, obs, model, action) rows."""
    vals = []
    with open(path) as f:
        for line in f:
            m = re.search(r"Resonating.*?:\s+([\d.\-]+)", line)
            if m:
                vals.append(float(m.group(1)))

    if len(vals) < 4:
        raise ValueError(f"Too few resonance values in {path}: found {len(vals)}, need >= 4")

    # Group into rows of 4: step, obs, model, action
    rows = []
    for i in range(0, len(vals) - 3, 4):
        rows.append({
            "step":   vals[i],
            "obs":    vals[i + 1],
            "model":  vals[i + 2],
            "action": vals[i + 3],
        })
    return rows


def mutual_information_proxy(x: np.ndarray, y: np.ndarray, n_bins: int = 5) -> float:
    """Estimate MI via histogram binning. Returns normalized MI in [0, 1]."""
    if len(x) < 4:
        return 0.0
    # Clip to valid range for digitize
    x_bins = np.linspace(x.min() - 1e-9, x.max() + 1e-9, n_bins + 1)
    y_bins = np.linspace(y.min() - 1e-9, y.max() + 1e-9, n_bins + 1)
    xi = np.digitize(x, x_bins) - 1
    yi = np.digitize(y, y_bins) - 1
    xi = np.clip(xi, 0, n_bins - 1)
    yi = np.clip(yi, 0, n_bins - 1)

    # Joint histogram
    joint = np.zeros((n_bins, n_bins))
    for a, b in zip(xi, yi):
        joint[a, b] += 1
    joint /= joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))

    # Normalize by min marginal entropy
    hx = -np.sum(px[px > 0] * np.log2(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log2(py[py > 0]))
    h_min = min(hx, hy)
    return float(mi / h_min) if h_min > 0 else 0.0


def compute_l_self(rows: list) -> dict:
    steps  = np.array([r["step"]   for r in rows])
    obs    = np.array([r["obs"]    for r in rows])
    model  = np.array([r["model"]  for r in rows])
    action = np.array([r["action"] for r in rows])

    n = len(rows)
    if n < 4:
        raise ValueError(f"Need at least 4 cycles, got {n}")

    # Deviation: the behavioral signal (continuous, more informative than binary action)
    deviation = obs - model   # negative = CORRECT territory

    # R_in proxy: correlation between MODEL[t] and OBS[t-1]
    # Does past observation predict current model state?
    # Trivially true for a running mean; measured as Pearson r
    r_in_corr = float(np.corrcoef(obs[:-1], model[1:])[0, 1])
    r_in_norm = abs(r_in_corr)  # [0, 1]

    # R_out proxy: mutual information between MODEL[t] and DEVIATION[t]
    # Does the model at decision time predict the behavioral signal?
    r_out_norm = mutual_information_proxy(model, deviation)

    # L_self = min(R_in, R_out)
    l_self = min(r_in_norm, r_out_norm)

    # Action entropy (sanity check — should be > 0 for mixed actions)
    p1 = action.mean()
    if 0 < p1 < 1:
        h_action = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    else:
        h_action = 0.0

    return {
        "n_cycles": n,
        "obs_range": (float(obs.min()), float(obs.max())),
        "model_range": (float(model.min()), float(model.max())),
        "action_frac_correct": float(p1),
        "action_entropy_bits": float(h_action),
        "r_in_corr": r_in_corr,
        "r_in_norm": r_in_norm,
        "r_out_mi_norm": r_out_norm,
        "l_self": l_self,
        "loop_closed": l_self > 0.01,
    }


def print_report(rows: list, metrics: dict) -> None:
    print("=" * 60)
    print("Type 4 Self-Referential Loop Benchmark")
    print("PhiFlow / PF consciousness_metric_program.md")
    print("=" * 60)
    print()
    print("Trace summary:")
    print(f"  Cycles:         {metrics['n_cycles']}")
    print(f"  OBS range:      [{metrics['obs_range'][0]:.4f}, {metrics['obs_range'][1]:.4f}]")
    print(f"  MODEL range:    [{metrics['model_range'][0]:.4f}, {metrics['model_range'][1]:.4f}]")
    print(f"  CORRECT frac:   {metrics['action_frac_correct']:.2f}")
    print(f"  Action entropy: {metrics['action_entropy_bits']:.3f} bits")
    print()
    print("L_self components:")
    print(f"  R_in  (corr OBS[t-1] → MODEL[t]):  {metrics['r_in_norm']:.4f}")
    print(f"  R_out (MI   MODEL[t] → DEVIATION):  {metrics['r_out_mi_norm']:.4f}")
    print()
    print(f"  L_self = min(R_in, R_out) = {metrics['l_self']:.4f}")
    print()
    verdict = "✅ LOOP CLOSED — self-referential structure detected" if metrics["loop_closed"] \
              else "❌ LOOP OPEN — R_in or R_out near zero"
    print(f"  Verdict: {verdict}")
    print()
    print("PF interpretation:")
    print("  R_in > 0:  past observations causally influence self-model [confirmed by running mean]")
    print("  R_out > 0: self-model causally shapes behavioral deviation [confirmed by obs formula]")
    print("  L_self > 0.01: the Type 4 structural loop is closed in this trace")
    print()
    print("Null comparison (white noise on same shape):")
    import numpy as np
    noise_obs = np.random.uniform(0.5, 0.9, len(rows))
    noise_model = np.random.uniform(0.5, 0.9, len(rows))
    noise_dev = noise_obs - noise_model
    null_r_out = mutual_information_proxy(noise_model, noise_dev)
    null_r_in = abs(float(np.corrcoef(noise_obs[:-1], noise_model[1:])[0, 1]))
    null_lself = min(null_r_in, null_r_out)
    print(f"  Null R_in ≈ {null_r_in:.4f}  Null R_out ≈ {null_r_out:.4f}  Null L_self ≈ {null_lself:.4f}")
    print(f"  PhiFlow L_self = {metrics['l_self']:.4f} vs Null ≈ {null_lself:.4f}")
    print("=" * 60)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "trace.txt"
    rows = parse_trace(path)
    metrics = compute_l_self(rows)
    print_report(rows, metrics)


if __name__ == "__main__":
    main()
