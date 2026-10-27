"""Tests for scripts/soma_motor_report_consumer.py — the R5 reference
consumer. Fail-closed is the contract under test: malformed, stale, or
absent input must produce no report, never a fabricated value.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import soma_motor_report_consumer as c  # noqa: E402


def write_bus(path, events):
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def actuate(value, ts, channel="soma_motor"):
    return {"type": "actuate", "channel": channel, "value": value,
            "ts": ts, "source": "test", "id": "t"}


NOW = "2099-01-01T00:00:00+00:00"  # far future → always within --max-age
OLD = "2020-01-01T00:00:00+00:00"  # stale


def test_valid_bus_line_writes_report(tmp_path):
    bus, state = tmp_path / "MOTOR.jsonl", tmp_path / "soma.json"
    write_bus(bus, [actuate(0.42, NOW)])
    v, _age = c.latest_bus_value(bus, "soma_motor", 1e9)
    assert v == 0.42
    c.write_report(state, v)
    s = json.loads(state.read_text())
    assert s["sensors"]["soma_motor_report"] == 0.42
    # complete SomaState envelope — the file must deserialize in Rust,
    # which requires every top-level field
    for k in ("schema_version", "session_id", "updated_at",
              "runtime", "health", "sensors"):
        assert k in s
    for k in ("sensor_stack", "ring_sensor_type",
              "sample_rate_hz", "fusion_interval_hz"):
        assert k in s["runtime"]
    assert s["health"]["fresh"] is True
    assert "baseline_locked" in s["health"]


def test_malformed_lines_skipped_newest_valid_wins(tmp_path):
    bus = tmp_path / "MOTOR.jsonl"
    bus.write_text(
        "not json\n"
        + json.dumps(actuate("NaN-str", NOW)) + "\n"   # bad value
        + json.dumps(actuate(0.5, NOW)) + "\n"          # good
        + json.dumps(actuate("bad", NOW)) + "\n"        # bad shadows nothing
    )
    v, _ = c.latest_bus_value(bus, "soma_motor", 1e9)
    assert v == 0.5


def test_wrong_channel_and_type_ignored(tmp_path):
    bus = tmp_path / "MOTOR.jsonl"
    write_bus(bus, [
        {"type": "other", "channel": "soma_motor", "value": 1, "ts": NOW},
        actuate(2.0, NOW, channel="other_chan"),
    ])
    assert c.latest_bus_value(bus, "soma_motor", 1e9) == (None, None)


def test_stale_event_fails_closed(tmp_path):
    bus = tmp_path / "MOTOR.jsonl"
    write_bus(bus, [actuate(9.9, OLD)])
    v, age = c.latest_bus_value(bus, "soma_motor", 10.0)
    assert v is None
    assert age is not None and age > 10.0


def test_missing_bus_absent(tmp_path):
    assert c.latest_bus_value(tmp_path / "nope.jsonl",
                              "soma_motor", 10.0) == (None, None)


def test_unparseable_ts_absent(tmp_path):
    bus = tmp_path / "MOTOR.jsonl"
    write_bus(bus, [actuate(1.0, "not-a-date")])
    assert c.latest_bus_value(bus, "soma_motor", 1e9) == (None, None)


def test_write_report_preserves_existing_fields(tmp_path):
    state = tmp_path / "soma.json"
    state.write_text(json.dumps({
        "schema_version": "soma.phiflow.v1",
        "updated_at": "old", "session_id": "real-producer",
        "runtime": {"sensor_stack": "SOMA", "ring_sensor_type": "CPU",
                    "sample_rate_hz": 500.0, "fusion_interval_hz": 5.0},
        "health": {"fresh": True, "age_ms": 5, "baseline_locked": True},
        "sensors": {"soma_presence": 0.9}}))
    c.write_report(state, 0.3)
    s = json.loads(state.read_text())
    assert s["sensors"]["soma_motor_report"] == 0.3
    assert s["sensors"]["soma_presence"] == 0.9        # preserved
    assert s["session_id"] == "real-producer"          # preserved
    assert s["runtime"]["sample_rate_hz"] == 500.0     # preserved


def test_constant_control_tick(tmp_path, monkeypatch):
    """--constant is the negative-control harness: it must write the
    constant regardless of bus state (no bus at all here)."""
    import types
    args = types.SimpleNamespace(constant=0.25, bus_path=tmp_path / "no.bus",
                                 channel="soma_motor", max_age=10.0,
                                 state_path=tmp_path / "soma.json")
    assert c.tick(args) == "constant"
    assert json.loads(args.state_path.read_text())[
        "sensors"]["soma_motor_report"] == 0.25


def test_dead_bus_tick_writes_nothing(tmp_path):
    import types
    args = types.SimpleNamespace(constant=None, bus_path=tmp_path / "no.bus",
                                 channel="soma_motor", max_age=10.0,
                                 state_path=tmp_path / "soma.json")
    assert c.tick(args) == "absent"
    assert not args.state_path.exists()
