#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OBJ_RE = re.compile(r'OBJ-\d{8}-\d{3,}')


def _parse_iso(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(raw.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _hours_since(ts: str, now: datetime) -> float | None:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    return (now - dt).total_seconds() / 3600.0


def _read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _objective_id_from_payload_ref(payload_ref: str) -> str | None:
    m = OBJ_RE.search(payload_ref)
    return m.group(0) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description='Weekly SLA audit for objective flow.')
    parser.add_argument('--pending-ack-sla-hours', type=int, required=True)
    parser.add_argument('--in-progress-sla-hours', type=int, required=True)
    args = parser.parse_args()

    qsop_root = Path(__file__).resolve().parents[1]
    objectives_dir = qsop_root / 'mail' / 'objectives'
    acks_dir = qsop_root / 'mail' / 'acks'
    queue_file = qsop_root.parents[1] / 'mcp-message-bus' / 'queue.json'

    now = datetime.now(timezone.utc)
    failed = False

    objective_packets: dict[str, dict[str, Any]] = {}
    payload_to_obj: dict[str, str] = {}

    for path in sorted(objectives_dir.glob('*.json')):
        try:
            data = _read_json(path)
            if isinstance(data, dict) and isinstance(data.get('objective_id'), str):
                obj_id = data['objective_id']
                objective_packets[obj_id] = data
                if isinstance(data.get('payload_ref'), str):
                    payload_to_obj[data['payload_ref'].lower()] = obj_id
        except Exception:
            print(f'[SLA_BREACH] {path.name}: objective packet unreadable')
            failed = True

    acked_objectives: set[str] = set()
    for path in sorted(acks_dir.glob('*.json')):
        try:
            data = _read_json(path)
            if isinstance(data, dict) and isinstance(data.get('objective_id'), str):
                acked_objectives.add(data['objective_id'])
        except Exception:
            print(f'[SLA_BREACH] {path.name}: ack packet unreadable')
            failed = True

    if queue_file.exists():
        try:
            q = _read_json(queue_file)
            if isinstance(q, list):
                for msg in q:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get('status') != 'pending':
                        continue
                    ts = msg.get('ts')
                    age_h = _hours_since(ts, now) if isinstance(ts, str) else None
                    payload_ref = msg.get('payload_ref') if isinstance(msg.get('payload_ref'), str) else ''
                    obj_id = payload_to_obj.get(payload_ref.lower()) if payload_ref else None
                    if obj_id is None and payload_ref:
                        obj_id = _objective_id_from_payload_ref(payload_ref)
                    label = obj_id or msg.get('id') or 'unknown_message'
                    if age_h is None:
                        print(f'[SLA_BREACH] {label}: pending queue message has invalid ts')
                        failed = True
                        continue
                    if age_h > args.pending_ack_sla_hours:
                        print(f'[SLA_BREACH] {label}: pending queue age {age_h:.2f}h > {args.pending_ack_sla_hours}h')
                        failed = True
                    else:
                        print(f'[OK] {label}: pending queue age {age_h:.2f}h <= {args.pending_ack_sla_hours}h')
        except Exception as e:
            print(f'[SLA_BREACH] queue.json: unreadable ({e})')
            failed = True
    else:
        print('[SLA_BREACH] queue.json: not found')
        failed = True

    for obj_id, data in sorted(objective_packets.items()):
        if obj_id in acked_objectives:
            print(f'[OK] {obj_id}: corresponding ack exists')
            continue
        ts = data.get('ts_utc')
        age_h = _hours_since(ts, now) if isinstance(ts, str) else None
        if age_h is None:
            print(f'[SLA_BREACH] {obj_id}: invalid or missing ts_utc')
            failed = True
            continue
        if age_h > args.in_progress_sla_hours:
            print(f'[SLA_BREACH] {obj_id}: no ack age {age_h:.2f}h > {args.in_progress_sla_hours}h')
            failed = True
        else:
            print(f'[OK] {obj_id}: no ack age {age_h:.2f}h <= {args.in_progress_sla_hours}h')

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
