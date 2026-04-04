#!/usr/bin/env python3
"""Weekly audit for QSOP freshness, packet validity, and objective metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys

from qsop_packet_lib import (
    load_json_files,
    parse_timestamp,
    qsop_root,
    validate_ack_packet,
    validate_objective_packet,
)

DEFAULT_PENDING_ACK_SLA_HOURS = 24.0
DEFAULT_IN_PROGRESS_SLA_HOURS = 48.0


def _days_since(date_value: datetime) -> int:
    now = datetime.now(timezone.utc)
    return int((now - date_value).total_seconds() // 86400)


def _parse_state_date(state_path: Path) -> datetime | None:
    first = state_path.read_text(encoding="utf-8").splitlines()[0]
    match = re.search(r"Last updated:\s*(\d{4}-\d{2}-\d{2})", first)
    if not match:
        return None
    return datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc)


def _latest_changelog_date(changelog_path: Path) -> datetime | None:
    content = changelog_path.read_text(encoding="utf-8")
    dates = re.findall(r"^##\s+(\d{4}-\d{2}-\d{2})", content, flags=re.MULTILINE)
    if not dates:
        return None
    return datetime.fromisoformat(dates[0]).replace(tzinfo=timezone.utc)


def _run_script(script: Path, root: Path, extra_args: list[str] | None = None) -> tuple[int, str]:
    args = [sys.executable, str(script)]
    if extra_args:
        args.extend(extra_args)
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _collect_sla_warnings(
    root: Path,
    pending_ack_sla_hours: float,
    in_progress_sla_hours: float,
) -> list[str]:
    warnings: list[str] = []
    objectives_dir = root / "mail" / "objectives"
    acks_dir = root / "mail" / "acks"

    objective_packets = load_json_files(objectives_dir)
    ack_packets = load_json_files(acks_dir)

    valid_objectives: dict[str, dict] = {}
    for path, packet in objective_packets:
        if not validate_objective_packet(path, packet):
            valid_objectives[packet["objective_id"]] = packet

    valid_acks_by_objective: dict[str, list[dict]] = defaultdict(list)
    for path, packet in ack_packets:
        if not validate_ack_packet(path, packet):
            valid_acks_by_objective[packet["objective_id"]].append(packet)

    now = datetime.now(timezone.utc)
    for objective_id, objective in sorted(valid_objectives.items()):
        created_at = parse_timestamp(objective["created_at"], "created_at")
        objective_age_hours = (now - created_at).total_seconds() / 3600.0
        objective_acks = sorted(
            valid_acks_by_objective.get(objective_id, []),
            key=lambda packet: parse_timestamp(packet["updated_at"], "updated_at"),
        )
        if not objective_acks:
            if objective_age_hours > pending_ack_sla_hours:
                warnings.append(
                    "SLA breach: objective has no ack "
                    f"after {objective_age_hours:.1f}h (threshold {pending_ack_sla_hours:.1f}h): {objective_id}"
                )
            continue

        latest_ack = objective_acks[-1]
        latest_status = latest_ack["status"]
        if latest_status == "in_progress":
            updated_at = parse_timestamp(latest_ack["updated_at"], "updated_at")
            ack_age_hours = (now - updated_at).total_seconds() / 3600.0
            if ack_age_hours > in_progress_sla_hours:
                warnings.append(
                    "SLA breach: in_progress objective has stale ack "
                    f"for {ack_age_hours:.1f}h (threshold {in_progress_sla_hours:.1f}h): {objective_id}"
                )
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run weekly QSOP audit.")
    parser.add_argument(
        "--pending-ack-sla-hours",
        type=float,
        default=DEFAULT_PENDING_ACK_SLA_HOURS,
        help=f"SLA for objective ACK creation (default: {DEFAULT_PENDING_ACK_SLA_HOURS}h).",
    )
    parser.add_argument(
        "--in-progress-sla-hours",
        type=float,
        default=DEFAULT_IN_PROGRESS_SLA_HOURS,
        help=f"SLA for in-progress ACK freshness (default: {DEFAULT_IN_PROGRESS_SLA_HOURS}h).",
    )
    args = parser.parse_args()

    root = qsop_root()
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    issues: list[str] = []
    warnings: list[str] = []

    state_path = root / "STATE.md"
    state_date = _parse_state_date(state_path) if state_path.exists() else None
    if state_date is None:
        issues.append("STATE.md missing valid `Last updated: YYYY-MM-DD` header on line 1.")
    else:
        state_age = _days_since(state_date)
        if state_age > 14:
            warnings.append(f"STATE.md is stale ({state_age} days old).")

    changelog_path = root / "CHANGELOG.md"
    change_date = _latest_changelog_date(changelog_path) if changelog_path.exists() else None
    if change_date is None:
        issues.append("CHANGELOG.md has no dated headings (`## YYYY-MM-DD ...`).")
    else:
        changelog_age = _days_since(change_date)
        if changelog_age > 14:
            warnings.append(f"CHANGELOG.md is stale ({changelog_age} days since latest entry).")

    validator_script = root / "tools" / "validate_packets.py"
    validator_exit, validator_output = _run_script(validator_script, root)
    if validator_exit != 0:
        issues.append("Packet validation failed.")

    warnings.extend(
        _collect_sla_warnings(
            root=root,
            pending_ack_sla_hours=args.pending_ack_sla_hours,
            in_progress_sla_hours=args.in_progress_sla_hours,
        )
    )

    metrics_file = metrics_dir / "objective_metrics.json"
    if not metrics_file.exists():
        warnings.append("Metrics file missing: run `log_objective_metrics.py`.")

    now_stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    report_path = metrics_dir / f"weekly_audit_{now_stamp}.md"
    lines = [
        "# Weekly QSOP Audit",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- QSOP root: `{root}`",
        f"- Pending ACK SLA (hours): {args.pending_ack_sla_hours:.1f}",
        f"- In-progress ACK SLA (hours): {args.in_progress_sla_hours:.1f}",
        "",
        "## Issues",
    ]
    if issues:
        lines.extend([f"- {item}" for item in issues])
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Warnings")
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Packet Validator Output")
    lines.append("```text")
    lines.append(validator_output or "(no output)")
    lines.append("```")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote audit report: {report_path}")

    if issues:
        print(f"Audit failed with {len(issues)} issue(s).")
        return 1
    print(f"Audit completed with {len(warnings)} warning(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
