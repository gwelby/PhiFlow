#!/usr/bin/env python3
"""Validate objective and ack packets in QSOP/mail."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from qsop_packet_lib import (
    ValidationIssue,
    load_json_files,
    qsop_root,
    validate_ack_packet,
    validate_objective_packet,
    verify_objective_payload_checksum,
)


def _render_issues(issues: list[ValidationIssue]) -> None:
    for issue in issues:
        print(f"ERROR: {issue.render()}")


def run_validation(mail_dir: Path, skip_checksum: bool = False) -> int:
    objectives_dir = mail_dir / "objectives"
    acks_dir = mail_dir / "acks"

    issues: list[ValidationIssue] = []

    objectives = load_json_files(objectives_dir)
    acks = load_json_files(acks_dir)

    objective_ids: set[str] = set()
    for path, packet in objectives:
        packet_issues = validate_objective_packet(path, packet)
        issues.extend(packet_issues)
        if not skip_checksum:
            issues.extend(verify_objective_payload_checksum(qsop_root(), path, packet))
        obj_id = packet.get("objective_id")
        if isinstance(obj_id, str):
            if obj_id in objective_ids:
                issues.append(ValidationIssue(path, f"duplicate `objective_id` detected: {obj_id}"))
            objective_ids.add(obj_id)

    for path, packet in acks:
        packet_issues = validate_ack_packet(path, packet)
        issues.extend(packet_issues)
        obj_id = packet.get("objective_id")
        if isinstance(obj_id, str) and obj_id not in objective_ids:
            issues.append(ValidationIssue(path, f"`objective_id` has no matching objective packet: {obj_id}"))

    _render_issues(issues)
    if issues:
        print(f"Validation failed: {len(issues)} issue(s).")
        return 1

    print(
        "Validation passed:"
        f" {len(objectives)} objective packet(s), {len(acks)} ack packet(s), 0 issues."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate QSOP objective and ack packets.")
    parser.add_argument(
        "--mail-dir",
        type=Path,
        default=qsop_root() / "mail",
        help="Path to mail directory (default: QSOP/mail).",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip objective payload checksum verification.",
    )
    args = parser.parse_args()
    return run_validation(args.mail_dir.resolve(), skip_checksum=args.skip_checksum)


if __name__ == "__main__":
    sys.exit(main())
