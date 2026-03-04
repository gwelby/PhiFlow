#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_iso(ts: str) -> bool:
    try:
        datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return True
    except Exception:
        return False


def _line(status: str, path: Path, reason: str) -> None:
    print(f'[{status}] {path.name}: {reason}')


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _validate_objective(data: dict[str, Any]) -> tuple[bool, str]:
    required = [
        'objective_id',
        'ts_utc',
        'from',
        'to',
        'intent',
        'payload_ref',
        'requires_ack',
    ]
    missing = [k for k in required if k not in data]
    if missing:
        return False, 'missing fields: ' + ', '.join(missing)

    if not isinstance(data['objective_id'], str) or not data['objective_id'].strip():
        return False, 'objective_id must be a non-empty string'
    if not isinstance(data['ts_utc'], str) or not _parse_iso(data['ts_utc']):
        return False, 'ts_utc must be an ISO timestamp'
    if not isinstance(data['from'], str) or not data['from'].strip():
        return False, 'from must be a non-empty string'
    if not isinstance(data['to'], str) or not data['to'].strip():
        return False, 'to must be a non-empty string'
    if not isinstance(data['intent'], str) or not data['intent'].strip():
        return False, 'intent must be a non-empty string'
    if not isinstance(data['payload_ref'], str) or not data['payload_ref'].strip():
        return False, 'payload_ref must be a non-empty string'
    if not isinstance(data['requires_ack'], bool):
        return False, 'requires_ack must be boolean'

    return True, 'ok'


def _validate_ack(data: dict[str, Any]) -> tuple[bool, str]:
    legacy = ['objective_id', 'message_id', 'agent_name', 'state', 'summary', 'evidence']
    modern = ['ack_id', 'objective_id', 'status', 'summary', 'verification']

    has_legacy = all(k in data for k in legacy)
    has_modern = all(k in data for k in modern)

    if not has_legacy and not has_modern:
        return False, 'missing required fields for legacy or modern ack schema'

    if has_legacy:
        if not isinstance(data['objective_id'], str) or not data['objective_id'].strip():
            return False, 'legacy objective_id must be a non-empty string'
        if not isinstance(data['message_id'], str) or not data['message_id'].strip():
            return False, 'legacy message_id must be a non-empty string'
        if not isinstance(data['agent_name'], str) or not data['agent_name'].strip():
            return False, 'legacy agent_name must be a non-empty string'
        if not isinstance(data['state'], str) or not data['state'].strip():
            return False, 'legacy state must be a non-empty string'
        if not isinstance(data['summary'], str) or not data['summary'].strip():
            return False, 'legacy summary must be a non-empty string'
        if not isinstance(data['evidence'], list):
            return False, 'legacy evidence must be a list'
        return True, 'ok (legacy schema)'

    if not isinstance(data['ack_id'], str) or not data['ack_id'].strip():
        return False, 'modern ack_id must be a non-empty string'
    if not isinstance(data['objective_id'], str) or not data['objective_id'].strip():
        return False, 'modern objective_id must be a non-empty string'
    if not isinstance(data['status'], str) or not data['status'].strip():
        return False, 'modern status must be a non-empty string'
    if not isinstance(data['summary'], str) or not data['summary'].strip():
        return False, 'modern summary must be a non-empty string'
    if not isinstance(data['verification'], (dict, list)):
        return False, 'modern verification must be object or list'

    status = str(data['status']).strip().lower()
    if status == 'completed':
        ok, reason = _validate_completed_ack_evidence(data['verification'])
        if not ok:
            return False, reason

    return True, 'ok (modern schema)'


def _looks_like_command(text: str) -> bool:
    # Broad command tokens across current PhiFlow lanes.
    return bool(
        re.search(
            r'(cargo|pytest|python3?|node|npm|wsl|bash|pwsh|run_all\.py|validate_packets\.py|phic|run_phi\.py)',
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_result(text: str) -> bool:
    return bool(
        re.search(
            r'(EXIT(?:_CODE)?\s*=?\s*\d+|PASS|FAILED|test result|ok|RITUAL COMPLETE|\bpassed\b|\bfailed\b)',
            text,
            flags=re.IGNORECASE,
        )
    )


def _validate_completed_ack_evidence(verification: Any) -> tuple[bool, str]:
    if not isinstance(verification, dict):
        return False, 'completed modern ACK requires verification object'

    if not verification:
        return False, 'completed modern ACK verification must not be empty'

    evidence_items: list[tuple[str, str]] = []
    for key, value in verification.items():
        if not isinstance(value, str) or not value.strip():
            return False, f'completed modern ACK verification[{key}] must be a non-empty string'
        evidence_items.append((str(key), value.strip()))

    # Truth gate: at least one line that looks like command output evidence.
    has_command_result = False
    for key, value in evidence_items:
        text = f'{key} {value}'
        has_exec_ref = _looks_like_command(text) or bool(
            re.search(
                r'(test|phase|run|regression|ritual|suite|acceptance|fail_first|post_fix)',
                key,
                flags=re.IGNORECASE,
            )
        )
        if has_exec_ref and _looks_like_result(value):
            has_command_result = True
            break

    if not has_command_result:
        return (
            False,
            'completed modern ACK requires at least one execution+result verification line',
        )

    return True, 'ok'


def main() -> int:
    qsop_root = Path(__file__).resolve().parents[1]
    objectives_dir = qsop_root / 'mail' / 'objectives'
    acks_dir = qsop_root / 'mail' / 'acks'

    failed = False

    for path in sorted(objectives_dir.glob('*.json')):
        data = _read_json(path)
        if data is None:
            _line('FAIL', path, 'invalid JSON or top-level not object')
            failed = True
            continue
        ok, reason = _validate_objective(data)
        _line('PASS' if ok else 'FAIL', path, reason)
        if not ok:
            failed = True

    for path in sorted(acks_dir.glob('*.json')):
        data = _read_json(path)
        if data is None:
            _line('FAIL', path, 'invalid JSON or top-level not object')
            failed = True
            continue
        ok, reason = _validate_ack(data)
        _line('PASS' if ok else 'FAIL', path, reason)
        if not ok:
            failed = True

    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
