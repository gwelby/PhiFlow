#!/usr/bin/env python3
"""Compute and persist objective metrics from QSOP objective/ack packets."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

from qsop_packet_lib import (
    load_json_files,
    parse_timestamp,
    qsop_root,
    validate_ack_packet,
    validate_objective_packet,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def main() -> int:
    root = qsop_root()
    objectives_dir = root / "mail" / "objectives"
    acks_dir = root / "mail" / "acks"
    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    objective_packets = load_json_files(objectives_dir)
    ack_packets = load_json_files(acks_dir)

    invalid_objectives: list[str] = []
    invalid_acks: list[str] = []

    objectives_by_id: dict[str, dict] = {}
    for path, packet in objective_packets:
        issues = validate_objective_packet(path, packet)
        if issues:
            invalid_objectives.append(str(path))
            continue
        obj_id = packet["objective_id"]
        objectives_by_id[obj_id] = packet

    acks_by_objective: dict[str, list[dict]] = defaultdict(list)
    for path, packet in ack_packets:
        issues = validate_ack_packet(path, packet)
        if issues:
            invalid_acks.append(str(path))
            continue
        acks_by_objective[packet["objective_id"]].append(packet)

    completed_lead_hours: list[float] = []
    completed_count = 0
    blocked_count = 0
    reopened_count = 0
    verification_covered_count = 0
    no_ack_objectives: list[str] = []

    for obj_id, objective in objectives_by_id.items():
        created_at = parse_timestamp(objective["created_at"], "created_at")
        acks = sorted(
            acks_by_objective.get(obj_id, []),
            key=lambda packet: parse_timestamp(packet["updated_at"], "updated_at"),
        )
        if not acks:
            no_ack_objectives.append(obj_id)
            continue

        status_seq = [ack["status"] for ack in acks]
        last_status = status_seq[-1]
        if last_status == "completed":
            completed_count += 1
            completed_at = parse_timestamp(acks[-1]["updated_at"], "updated_at")
            delta_hours = (completed_at - created_at).total_seconds() / 3600.0
            completed_lead_hours.append(delta_hours)
        if last_status == "blocked":
            blocked_count += 1
        if "completed" in status_seq and any(s in {"in_progress", "blocked"} for s in status_seq[status_seq.index("completed") + 1 :]):
            reopened_count += 1

        has_verification = any(bool(ack.get("verification")) for ack in acks)
        if has_verification:
            verification_covered_count += 1

    total_objectives = len(objectives_by_id)
    acked_objectives = total_objectives - len(no_ack_objectives)
    avg_lead_hours = statistics.fmean(completed_lead_hours) if completed_lead_hours else None

    metrics = {
        "generated_at": _now_iso(),
        "counts": {
            "total_objectives": total_objectives,
            "acked_objectives": acked_objectives,
            "completed_objectives": completed_count,
            "blocked_objectives": blocked_count,
            "reopened_objectives": reopened_count,
        },
        "rates": {
            "ack_coverage": (acked_objectives / total_objectives) if total_objectives else 0.0,
            "verification_coverage": (verification_covered_count / total_objectives) if total_objectives else 0.0,
            "reopen_rate": (reopened_count / total_objectives) if total_objectives else 0.0,
        },
        "timing": {
            "average_lead_time_hours_for_completed": avg_lead_hours,
            "completed_lead_time_hours": completed_lead_hours,
        },
        "invalid_packets": {
            "objectives": invalid_objectives,
            "acks": invalid_acks,
        },
        "no_ack_objectives": no_ack_objectives,
    }

    metrics_path = metrics_dir / "objective_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    print(f"Wrote metrics: {metrics_path}")
    print(
        "Summary:"
        f" total={total_objectives}, acked={acked_objectives},"
        f" completed={completed_count}, blocked={blocked_count}, reopened={reopened_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
