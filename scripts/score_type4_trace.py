#!/usr/bin/env python3
"""score_type4_trace.py — independent Type-4 self-correlation scorer.

Re-implements the canonical Rust math
(`src/metrics/self_correlation.rs::from_type4_trace`) in pure Python so a
second seat can recompute L_self from a raw trace file WITHOUT trusting the
experimenters' phic binary — R5 prereg Amendment A, item 3 (independent
scorer).

Numbers should match the Rust implementation to float noise. A divergence
is itself a finding: report it, do not reconcile silently — that would be
the instrument lying in stereo.

Trace format (type4_trace_benchmark): one numeric value per line as
"Resonating <name>: <value>" (bare numerics also accepted), grouped in
4-tuples: step, obs, model, action.

Outputs: JSON + human summary. Also reports the R5 diagnostics declared in
the prereg: corr(action[t], obs[t+1]) — the engineered-edge signal check.

Usage:
    python3 scripts/score_type4_trace.py tests/fixtures/soma_live_trace.txt
    python3 scripts/score_type4_trace.py trace.txt --threshold 0.1 --json
"""

import json
import math
import sys


def parse_trace(path):
    """Parse trace file -> list of (step, obs, model, action) tuples."""
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            pos = line.find("Resonating")
            if pos >= 0:
                colon = line.find(":", pos)
                if colon >= 0:
                    tok = line[colon + 1:].split()
                    if tok:
                        try:
                            vals.append(float(tok[0]))
                            continue
                        except ValueError:
                            pass
            else:
                try:
                    vals.append(float(line))
                except ValueError:
                    pass
    return [tuple(vals[i:i + 4]) for i in range(0, len(vals) - 3, 4)]


def pearson(x, y):
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = den_x = den_y = 0.0
    for xi, yi in zip(x, y):
        dx, dy = xi - mx, yi - my
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def _bin(v, lo, rng, bins):
    return int(min(max((v - lo) / rng * bins, 0.0), bins - 1.0))


def entropy(samples, bins):
    if len(samples) < 2 or bins == 0:
        return 0.0
    lo, hi = min(samples), max(samples)
    if hi <= lo:
        return 0.0
    rng = max(hi - lo, 1e-10)
    counts = {}
    for v in samples:
        b = _bin(v, lo, rng, bins)
        counts[b] = counts.get(b, 0) + 1
    n = len(samples)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def mi_from_samples(x, y, bins):
    n = len(x)
    if n != len(y) or n < 2 or bins == 0:
        return 0.0
    lx, hx = min(x), max(x)
    ly, hy = min(y), max(y)
    if hx <= lx or hy <= ly:
        return 0.0
    rx = max(hx - lx, 1e-10)
    ry = max(hy - ly, 1e-10)
    joint, mx_, my_ = {}, {}, {}
    for xi, yi in zip(x, y):
        bx = _bin(xi, lx, rx, bins)
        by = _bin(yi, ly, ry, bins)
        joint[(bx, by)] = joint.get((bx, by), 0.0) + 1.0 / n
        mx_[bx] = mx_.get(bx, 0.0) + 1.0 / n
        my_[by] = my_.get(by, 0.0) + 1.0 / n
    mi = 0.0
    for (bx, by), pxy in joint.items():
        if pxy > 0:
            mi += pxy * math.log2(pxy / (mx_[bx] * my_[by]))
    return mi


def normalized_mi(x, y, bins):
    mi = mi_from_samples(x, y, bins)
    h_min = min(entropy(x, bins), entropy(y, bins))
    return max(0.0, min(1.0, mi / h_min)) if h_min > 0 else 0.0


def score(path, threshold=0.1, bins=5):
    tuples = parse_trace(path)
    if len(tuples) < 4:
        return {"error": f"trace too short ({len(tuples)} tuples, need >= 4)",
                "tuples": len(tuples)}
    obs = [t[1] for t in tuples]
    model = [t[2] for t in tuples]
    action = [t[3] for t in tuples]

    r_in = abs(pearson(obs[:-1], model[1:]))
    r_out = normalized_mi(model[:-1], action[1:], bins)
    l_self = min(r_in, r_out)
    # R5 engineered-edge diagnostic: does action[t] reach obs[t+1]?
    corr_act_obs = pearson(action[:-1], obs[1:])

    return {
        "trace": path,
        "cycles": len(tuples),
        "r_in": round(r_in, 6),
        "r_out": round(r_out, 6),
        "l_self": round(l_self, 6),
        "threshold": threshold,
        "loop_closed": l_self > threshold,
        "corr_action_obsnext": round(corr_act_obs, 6),
        "scorer": "score_type4_trace.py (independent Python reimpl of "
                  "self_correlation::from_type4_trace)",
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(2)
    threshold = 0.1
    if "--threshold" in sys.argv:
        i = sys.argv.index("--threshold")
        threshold = float(sys.argv[i + 1])
    res = score(args[0], threshold)
    if "--json" in sys.argv:
        print(json.dumps(res, indent=2))
    else:
        for k, v in res.items():
            print(f"  {k:22} {v}")
        if "l_self" in res:
            verdict = "CLOSED" if res["loop_closed"] else "OPEN"
            print(f"\n  VERDICT: {verdict} (L_self {res['l_self']} vs {threshold})")


if __name__ == "__main__":
    main()
