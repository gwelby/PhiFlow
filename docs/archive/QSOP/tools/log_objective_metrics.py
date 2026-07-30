#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def main() -> int:
    parser = argparse.ArgumentParser(description='Append objective metrics snapshot.')
    parser.add_argument('--objective-id', required=True)
    parser.add_argument('--ast-resonance', type=float)
    parser.add_argument('--host-stability', type=float)
    parser.add_argument('--mutation-depth', type=float)
    parser.add_argument('--evidence-coherence', type=float)
    args = parser.parse_args()

    qsop_root = Path(__file__).resolve().parents[1]
    metrics_dir = qsop_root / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = metrics_dir / 'metrics_log.jsonl'

    record = {
        'ts': _iso_now(),
        'objective_id': args.objective_id,
        'ast_resonance': args.ast_resonance,
        'host_stability': args.host_stability,
        'mutation_depth': args.mutation_depth,
        'evidence_coherence': args.evidence_coherence,
    }

    with metrics_log.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

    print(f'WROTE {metrics_log}')
    print(json.dumps(record))
    return 0


if __name__ == '__main__':
    sys.exit(main())
