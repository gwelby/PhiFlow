#!/usr/bin/env python3
"""Run full QSOP tooling chain in one command."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from qsop_packet_lib import qsop_root


def _run(script: Path, extra_args: list[str], cwd: Path) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"==> Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=False)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validate, metrics, and weekly audit in sequence.")
    parser.add_argument(
        "--pending-ack-sla-hours",
        type=float,
        default=24.0,
        help="SLA for objective ACK creation in hours (passed to weekly audit).",
    )
    parser.add_argument(
        "--in-progress-sla-hours",
        type=float,
        default=48.0,
        help="SLA for in-progress ACK freshness in hours (passed to weekly audit).",
    )
    args = parser.parse_args()

    root = qsop_root()
    tools = root / "tools"

    steps = [
        ("validate_packets.py", []),
        ("log_objective_metrics.py", []),
        (
            "weekly_qsop_audit.py",
            [
                "--pending-ack-sla-hours",
                str(args.pending_ack_sla_hours),
                "--in-progress-sla-hours",
                str(args.in_progress_sla_hours),
            ],
        ),
    ]

    for script_name, extra_args in steps:
        code = _run(tools / script_name, extra_args=extra_args, cwd=root)
        if code != 0:
            print(f"FAILED: {script_name} exited with code {code}")
            return code

    print("All QSOP tooling steps passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
