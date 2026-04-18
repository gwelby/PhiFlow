#!/usr/bin/env python3
"""Shared helpers for QSOP objective/ack packet validation and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

OBJECTIVE_ID_RE = re.compile(r"^OBJ-\d{8}-\d{3,}$")
CHECKSUM_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ACK_ID_RE = re.compile(r"^ACK-")

OBJECTIVE_STATUSES = {"pending", "in_progress", "completed", "blocked", "cancelled"}
ACK_STATUSES = {"accepted", "in_progress", "completed", "blocked", "rejected"}


@dataclass
class ValidationIssue:
    path: Path
    message: str

    def render(self) -> str:
        return f"{self.path}: {self.message}"


def qsop_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_timestamp(raw: Any, field: str) -> datetime:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"field `{field}` must be a non-empty string timestamp")
    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as err:
        raise ValueError(f"field `{field}` must be ISO-8601, got `{raw}`") from err
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_json_files(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    if not directory.exists():
        return []
    packets: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            parsed = json.load(handle)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}: top-level JSON value must be an object")
        packets.append((path, parsed))
    return packets


def _require_string(packet: dict[str, Any], key: str, issues: list[ValidationIssue], path: Path) -> str | None:
    value = packet.get(key)
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue(path, f"missing/invalid string field `{key}`"))
        return None
    return value


def validate_objective_packet(path: Path, packet: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    objective_id = _require_string(packet, "objective_id", issues, path)
    if objective_id and not OBJECTIVE_ID_RE.match(objective_id):
        issues.append(ValidationIssue(path, f"`objective_id` has invalid format: {objective_id}"))

    for key in ("origin", "target", "intent", "payload_path"):
        _require_string(packet, key, issues, path)

    status = _require_string(packet, "status", issues, path)
    if status and status not in OBJECTIVE_STATUSES:
        issues.append(ValidationIssue(path, f"`status` must be one of {sorted(OBJECTIVE_STATUSES)}"))

    checksum = _require_string(packet, "checksum", issues, path)
    if checksum and not CHECKSUM_RE.match(checksum):
        issues.append(ValidationIssue(path, "`checksum` must be `sha256:<64 lowercase hex>`"))

    if "created_at" not in packet:
        issues.append(ValidationIssue(path, "missing required field `created_at`"))
    else:
        try:
            parse_timestamp(packet.get("created_at"), "created_at")
        except ValueError as err:
            issues.append(ValidationIssue(path, str(err)))

    ownership = packet.get("ownership")
    if ownership is not None:
        if not isinstance(ownership, list) or not all(isinstance(x, str) and x.strip() for x in ownership):
            issues.append(ValidationIssue(path, "`ownership` must be a list of non-empty strings"))

    verification_required = packet.get("verification_required")
    if verification_required is not None and not isinstance(verification_required, bool):
        issues.append(ValidationIssue(path, "`verification_required` must be boolean when present"))

    return issues


def validate_ack_packet(path: Path, packet: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    ack_id = _require_string(packet, "ack_id", issues, path)
    if ack_id and not ACK_ID_RE.match(ack_id):
        issues.append(ValidationIssue(path, "`ack_id` must start with `ACK-`"))

    objective_id = _require_string(packet, "objective_id", issues, path)
    if objective_id and not OBJECTIVE_ID_RE.match(objective_id):
        issues.append(ValidationIssue(path, f"`objective_id` has invalid format: {objective_id}"))

    _require_string(packet, "responder", issues, path)
    _require_string(packet, "summary", issues, path)

    status = _require_string(packet, "status", issues, path)
    if status and status not in ACK_STATUSES:
        issues.append(ValidationIssue(path, f"`status` must be one of {sorted(ACK_STATUSES)}"))

    if "updated_at" not in packet:
        issues.append(ValidationIssue(path, "missing required field `updated_at`"))
    else:
        try:
            parse_timestamp(packet.get("updated_at"), "updated_at")
        except ValueError as err:
            issues.append(ValidationIssue(path, str(err)))

    files_changed = packet.get("files_changed")
    if files_changed is not None:
        if not isinstance(files_changed, list) or not all(isinstance(x, str) for x in files_changed):
            issues.append(ValidationIssue(path, "`files_changed` must be a list of strings"))

    verification = packet.get("verification")
    if verification is not None:
        if not isinstance(verification, list):
            issues.append(ValidationIssue(path, "`verification` must be a list"))
        else:
            for idx, item in enumerate(verification):
                if not isinstance(item, dict):
                    issues.append(ValidationIssue(path, f"`verification[{idx}]` must be an object"))
                    continue
                if not isinstance(item.get("command"), str) or not item["command"].strip():
                    issues.append(ValidationIssue(path, f"`verification[{idx}].command` must be non-empty string"))
                if not isinstance(item.get("result"), str) or not item["result"].strip():
                    issues.append(ValidationIssue(path, f"`verification[{idx}].result` must be non-empty string"))

    risks = packet.get("risks")
    if risks is not None:
        if not isinstance(risks, list) or not all(isinstance(x, str) for x in risks):
            issues.append(ValidationIssue(path, "`risks` must be a list of strings"))

    return issues


def resolve_payload_path(qsop_root_path: Path, payload_path: str) -> Path:
    candidate = Path(payload_path)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0].lower() == "qsop":
        return (qsop_root_path.parent / candidate).resolve()
    return (qsop_root_path / candidate).resolve()


def compute_sha256_for_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_objective_payload_checksum(
    qsop_root_path: Path,
    objective_packet_path: Path,
    packet: dict[str, Any],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    payload_value = packet.get("payload_path")
    checksum_value = packet.get("checksum")
    if not isinstance(payload_value, str) or not payload_value.strip():
        return issues
    if not isinstance(checksum_value, str) or not CHECKSUM_RE.match(checksum_value):
        return issues

    payload_path = resolve_payload_path(qsop_root_path, payload_value)
    if not payload_path.exists() or not payload_path.is_file():
        issues.append(
            ValidationIssue(
                objective_packet_path,
                f"`payload_path` does not resolve to a file: {payload_path}",
            )
        )
        return issues

    expected = checksum_value.split(":", 1)[1]
    actual = compute_sha256_for_path(payload_path)
    if actual != expected:
        issues.append(
            ValidationIssue(
                objective_packet_path,
                f"payload checksum mismatch for {payload_path} (expected {expected}, actual {actual})",
            )
        )
    return issues
