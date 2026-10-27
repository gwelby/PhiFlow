#!/usr/bin/env python3
"""soma_motor_report_consumer.py — reference implementation of the R5
`soma_motor_report` consumer contract (prereg:
QSOP/EVIDENCE/type4_r5_actuate_prereg_2026-10-24.md).

Loop it closes:
    phic actuate motor("soma_motor", v) -> MOTOR.jsonl
    -> THIS consumer -> soma_state.json::sensors.soma_motor_report
    -> phic sensor("soma_motor_report") -> model

Contract (declared before it existed):
  - read the latest `actuate` event on the target channel from MOTOR.jsonl
  - write sensors.soma_motor_report = value into the SOMA state file,
    timestamped no older than one sample window
  - FAIL-CLOSED: no fresh bus event -> write nothing. A dead bus must read
    as absent (sensor -> None), never as a held value.

--constant V is the R5 negative control (Amendment A item 2): writes V
unconditionally, ignoring the bus. If the metric still crosses L_self > 0.1
under a constant report, the scorer is seeing program periodicity, not the
loop — that result invalidates the surface, not the language. Control runs
FIRST per the prereg.

NOTE: this process writes into a file the SOMA producer owns. On a live
SOMA host the producer overwrites state per sample, so the durable home
for this logic is inside the producer (AntiGravity lane) or a merge
sidecar. This reference implementation is also runnable standalone for
fixture-driven R5 runs (phic reads the same SOMA_STATE_PATH).

Usage:
    # one-shot against a live bus
    python3 scripts/soma_motor_report_consumer.py --once

    # daemon loop
    python3 scripts/soma_motor_report_consumer.py --interval 1.0

    # negative control — constant report, bus ignored
    python3 scripts/soma_motor_report_consumer.py --constant 0.5 --once
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def default_bus_path():
    return os.environ.get(
        "MOTOR_BUS_PATH",
        str(Path.home() / ".local/share/phiflow/MOTOR.jsonl"))


def default_state_path():
    return os.environ.get(
        "SOMA_STATE_PATH",
        str(Path.home() / ".local/share/phiflow/soma_state.json"))


def latest_bus_value(bus_path, channel, max_age_s):
    """Return (value, age_s) of the freshest actuate event on `channel`,
    or (None, None) if the bus is absent/empty/stale."""
    try:
        lines = Path(bus_path).read_text().splitlines()
    except OSError:
        return None, None
    newest = None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") != "actuate" or ev.get("channel") != channel:
            continue
        # A malformed event must not shadow a good one AND must not crash:
        # validate value (finite number) before accepting the line.
        try:
            value = float(ev["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        newest = (ev, value)
        break
    if newest is None:
        return None, None
    ev, value = newest
    try:
        ts = datetime.fromisoformat(str(ev["ts"]).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:
        return None, None
    if age > max_age_s:
        return None, age
    return value, age


def write_report(state_path, value):
    """Read-modify-write sensors.soma_motor_report atomically.

    On a live SOMA host the producer owns every field except
    soma_motor_report; on standalone/fixture runs the file may not exist,
    so we stamp the minimal envelope is_soma_state_fresh() requires —
    schema_version, updated_at, health.fresh. A real producer's per-sample
    rewrite simply overwrites these.
    """
    p = Path(state_path)
    try:
        state = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    now = datetime.now(timezone.utc).isoformat()
    # SomaState has no serde defaults at the top level: every field below is
    # REQUIRED for the file to deserialize at all. A partial runtime/health
    # object silently drops the whole state to None → sensor() reads 0.0.
    # setdefault preserves a live producer's fields; a scratch file gets a
    # complete, spec-valid envelope.
    state.setdefault("schema_version", "soma.phiflow.v1")
    state.setdefault("session_id", "motor-report-consumer")
    state["updated_at"] = now
    runtime = state.setdefault("runtime", {})
    runtime.setdefault("sensor_stack", "motor_bus")
    runtime.setdefault("ring_sensor_type", "none")
    runtime.setdefault("sample_rate_hz", 0.0)
    runtime.setdefault("fusion_interval_hz", 0.0)
    runtime["motor_report_ts"] = now
    health = state.setdefault("health", {})
    health["fresh"] = True
    health["age_ms"] = 0
    health.setdefault("baseline_locked", False)
    state.setdefault("sensors", {})["soma_motor_report"] = value
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, p)


def tick(args):
    if args.constant is not None:
        write_report(args.state_path, args.constant)
        return "constant"
    value, age = latest_bus_value(args.bus_path, args.channel, args.max_age)
    if value is None:
        # fail-closed: stale or silent bus -> write nothing
        return "absent"
    write_report(args.state_path, float(value))
    return f"wrote {value:.6f} (age {age:.1f}s)" if age else f"wrote {value}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus-path", default=default_bus_path())
    ap.add_argument("--state-path", default=default_state_path())
    ap.add_argument("--channel", default="soma_motor")
    ap.add_argument("--max-age", type=float, default=10.0,
                    help="bus events older than this many seconds count as absent")
    ap.add_argument("--constant", type=float, default=None,
                    help="negative control: write this value regardless of the bus")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0)
    args = ap.parse_args()

    if args.once or args.interval <= 0:
        print(tick(args))
        return
    while True:
        print(tick(args), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
