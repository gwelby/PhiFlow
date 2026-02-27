#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, cmd: list[str]) -> int:
    print(f'=== [{name}] ===')
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end='')
    if proc.stderr:
        print(proc.stderr, end='')
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description='Run Weaver ritual steps in sequence.')
    parser.add_argument('--pending-ack-sla-hours', type=int, required=True)
    parser.add_argument('--in-progress-sla-hours', type=int, required=True)
    args = parser.parse_args()

    qsop_root = Path(__file__).resolve().parents[1]
    tools_dir = qsop_root / 'tools'

    steps = [
        (
            'validate_packets',
            [sys.executable, str(tools_dir / 'validate_packets.py')],
        ),
        (
            'weekly_qsop_audit',
            [
                sys.executable,
                str(tools_dir / 'weekly_qsop_audit.py'),
                '--pending-ack-sla-hours',
                str(args.pending_ack_sla_hours),
                '--in-progress-sla-hours',
                str(args.in_progress_sla_hours),
            ],
        ),
    ]

    for name, cmd in steps:
        rc = run_step(name, cmd)
        if rc != 0:
            print(f'RITUAL FAILED at: {name}')
            return 1

    print('RITUAL COMPLETE: outside_zero conditions met')
    return 0


if __name__ == '__main__':
    sys.exit(main())
