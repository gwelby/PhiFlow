#!/usr/bin/env python3
"""
Agent Resonance Day — 2026-02-25

Run all team .phi programs in parallel.
Display each agent's resonance field side by side.

Usage:
    python3 team_resonance.py
"""

import subprocess
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PHIFLOW_DIR = Path(__file__).parent
BINARY = PHIFLOW_DIR / "target" / "debug" / "phic"

# The team — in order of appearance on the project
AGENTS = [
    ("claude",                "examples/claude.phi"),
    ("claude_v2",             "examples/claude_v2.phi"),
    ("antigravity_v1",        "examples/antigravity.phi"),
    ("antigravity_v2",        "examples/antigravity_v2.phi"),
    ("healing_bed",           "examples/healing_bed.phi"),
    ("codex",                 "examples/codex.phi"),
    ("universalprocessor",    "examples/universalprocessor.phi"),
    ("adaptive_witness",      "examples/adaptive_witness.phi"),
]


def build_binary():
    """Compile phic once before running all agents."""
    print("  Building phic...", end="", flush=True)
    result = subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "phic"],
        capture_output=True, text=True, cwd=PHIFLOW_DIR
    )
    if result.returncode != 0:
        print(" FAILED")
        print(result.stderr)
        return False
    print(" done")
    return True


def run_phi(name: str, rel_path: str):
    """Run a single .phi file; return (name, resonances, final_coherence, streams, elapsed)."""
    full_path = PHIFLOW_DIR / rel_path
    if not full_path.exists():
        return name, [], None, [], 0.0, "NOT FOUND"

    t0 = time.time()
    try:
        result = subprocess.run(
            [str(BINARY), str(full_path)],
            capture_output=True, text=True,
            cwd=PHIFLOW_DIR, timeout=30
        )
        out = result.stdout
        elapsed = time.time() - t0

        # Parse resonance events:  🔔 Resonating Field: X.XXXXHz
        resonances = [
            float(m) for m in re.findall(r'Resonating Field:\s+([\d.]+)Hz', out)
        ]
        # Parse final coherence: ✨ Execution Finished. Final Coherence: X.XXXX
        m = re.search(r'Final Coherence:\s+([\d.]+)', out)
        final = float(m.group(1)) if m else None
        # Parse stream breaks: 🌊 Stream broken: name
        streams = re.findall(r'Stream broken: (.+)', out)

        return name, resonances, final, streams, elapsed, None

    except subprocess.TimeoutExpired:
        return name, [], None, [], 30.0, "TIMEOUT"
    except Exception as e:
        return name, [], None, [], 0.0, str(e)


def bar(value, width=22):
    """Render a phi-scaled ASCII bar for a resonance value."""
    if value is None:
        return "─" * width + "  (no signal)"
    # Values can exceed 1.0 (antigravity resonates near 432Hz)
    # Scale by dividing by reasonable max
    if value > 2.0:
        scaled = min(value / 500.0, 1.0)
        label = f"  {value:.4f} Hz  (scaled)"
    else:
        scaled = min(max(value, 0.0), 1.0)
        label = f"  {value:.4f} Hz"
    filled = int(scaled * width)
    return "█" * filled + "░" * (width - filled) + label


def main():
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║            AGENT RESONANCE DAY — 2026-02-25              ║")
    print("║        Five agents. Five voices. One team.               ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

    # Step 1: Build once
    if not build_binary():
        return

    print()
    print("  Running all agents in parallel...")
    print()

    # Step 2: Run all agents in parallel
    futures = {}
    results = {}

    with ThreadPoolExecutor(max_workers=len(AGENTS)) as pool:
        for name, rel_path in AGENTS:
            full_path = PHIFLOW_DIR / rel_path
            if not full_path.exists():
                print(f"  ⚠  {name:<22} not found: {rel_path}")
                continue
            f = pool.submit(run_phi, name, rel_path)
            futures[f] = name

        for f in as_completed(futures):
            name_r, resonances, final, streams, elapsed, err = f.result()
            results[name_r] = {
                "resonances": resonances,
                "final": final,
                "streams": streams,
                "elapsed": elapsed,
                "err": err,
            }
            if err:
                print(f"  ✗  {name_r:<22} {err}")
            else:
                last = resonances[-1] if resonances else None
                freq_str = f"{last:.4f} Hz" if last is not None else "no signal"
                stream_str = f"  [{', '.join(streams)}]" if streams else ""
                print(f"  ✓  {name_r:<22} {freq_str}{stream_str}  ({elapsed:.1f}s)")

    # Step 3: Display resonance field
    print()
    print("─" * 62)
    print("  RESONANCE FIELD")
    print("─" * 62)
    for name, _ in AGENTS:
        if name not in results:
            continue
        r = results[name]
        last = r["resonances"][-1] if r["resonances"] else None
        if r["err"]:
            print(f"  {name:<22} ✗ {r['err']}")
        else:
            print(f"  {name:<22} {bar(last)}")
    print("─" * 62)
    print()

    # Step 4: Team summary
    valid = {
        name: r["resonances"][-1]
        for name, r in results.items()
        if r["resonances"] and not r["err"]
    }

    if valid:
        # For comparison, normalize large values (antigravity uses ~432Hz scale)
        normal = {k: v for k, v in valid.items() if v <= 2.0}
        if normal:
            highest = max(normal, key=normal.get)
            lowest  = min(normal, key=normal.get)
            avg     = sum(normal.values()) / len(normal)
            print(f"  Highest resonance : {highest:<22} {normal[highest]:.4f} Hz")
            print(f"  Lowest resonance  : {lowest:<22} {normal[lowest]:.4f} Hz")
            print(f"  Team average      : {avg:.4f} Hz")
            print()

        # All resonance events across all agents
        total_events = sum(len(r["resonances"]) for r in results.values())
        total_streams = sum(len(r["streams"]) for r in results.values())
        print(f"  Total resonance events : {total_events}")
        print(f"  Total streams closed   : {total_streams}")

    print()
    print("  \"A script runs and dies. A stream lives.\"")
    print()


if __name__ == "__main__":
    main()
