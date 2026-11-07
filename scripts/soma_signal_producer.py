#!/usr/bin/env python3
"""soma_signal_producer.py — ENGINEERED-SIGNAL state producer.

Prereg: QSOP/EVIDENCE/engineered_signal_lane_prereg_2026-11-07.md

Writes a complete, spec-valid SomaState envelope to SOMA_STATE_PATH at a
declared cadence with DECLARED waveform values — the engineered producer
that feeds the real sensor() path when the natural producer is absent.

EVERYTHING this producer emits is ENGINEERED-SIGNAL evidence. It exists
to exercise the instrument (freshness gate -> sensor() -> daemon -> trace
-> scorer), never to imitate nature. Nothing it produces may be pooled
with natural-trace evidence.

Default waveform (declared in the prereg):
  soma_presence   = 0.55 + 0.35*sin(2*pi*t/60s)   (0.2..0.9, 60s period)
  soma_schumann   = 0.5 + 0.25*sin(2*pi*t/47s)    (independent period so
                    channels don't move in lockstep — lockstep channels
                    fake correlation structure)
  soma_432        = 0.6 constant
  soma_peak_dbc   = -40.0 + 10*sin(2*pi*t/23s)
  ring_slope_1f   = -1.1 + 0.2*sin(2*pi*t/31s)
  ring_jitter_ns  = 3.0 + 1.0*sin(2*pi*t/19s)
  ring_phase_delta= 0.05 + 0.03*sin(2*pi*t/41s)
  soma_motor_report left untouched (None-absent preserved)

Modes:
  --wave           declared waveform (default)
  --constant V     every varying channel pinned to V — the lane's
                   negative control: a constant signal MUST produce a
                   degenerate trace (scorer degenerate:true) or the
                   instrument is scoring its own machinery.
  --drift          slow linear ramp (presence 0.2->0.9 over --duration)

Usage:
  python3 scripts/soma_signal_producer.py --state-path /tmp/soma_state.json \
      --interval 0.25 --duration 900
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

CHANNELS = (
    "soma_schumann", "soma_432", "soma_presence", "soma_fan_hz",
    "soma_ac_60", "soma_peak_dbc", "ring_slope_1f", "ring_jitter_ns",
    "ring_coherence_432", "ring_coherence_528", "ring_phase_delta",
)


def wave_values(t):
    """Declared waveform; t = seconds since producer start."""
    return {
        "soma_presence": 0.55 + 0.35 * math.sin(2 * math.pi * t / 60.0),
        "soma_schumann": 0.5 + 0.25 * math.sin(2 * math.pi * t / 47.0),
        "soma_432": 0.6,
        "soma_fan_hz": 0.0,
        "soma_ac_60": 0.0,
        "soma_peak_dbc": -40.0 + 10.0 * math.sin(2 * math.pi * t / 23.0),
        "ring_slope_1f": -1.1 + 0.2 * math.sin(2 * math.pi * t / 31.0),
        "ring_jitter_ns": 3.0 + 1.0 * math.sin(2 * math.pi * t / 19.0),
        "ring_coherence_432": 0.5,
        "ring_coherence_528": 0.5,
        "ring_phase_delta": 0.05 + 0.03 * math.sin(2 * math.pi * t / 41.0),
    }


def constant_values(v):
    return {c: v for c in CHANNELS}


def drift_values(t, duration):
    frac = min(max(t / max(duration, 1.0), 0.0), 1.0)
    vals = wave_values(t)
    vals["soma_presence"] = 0.2 + 0.7 * frac
    return vals


def write_state(state_path, sensors):
    """Complete spec-valid SomaState envelope — no serde defaults at the
    top level, so every required field is stamped every write. Atomic
    tmp+replace, same contract as the motor-report consumer."""
    p = Path(state_path)
    try:
        state = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    now = datetime.now(timezone.utc).isoformat()
    state.setdefault("schema_version", "soma.phiflow.v1")
    state.setdefault("session_id", "ENGINEERED-SIGNAL-producer")
    state["updated_at"] = now
    runtime = state.setdefault("runtime", {})
    runtime.setdefault("sensor_stack", "ENGINEERED-SIGNAL")
    runtime.setdefault("ring_sensor_type", "ENGINEERED-SIGNAL")
    runtime.setdefault("sample_rate_hz", 4.0)
    runtime.setdefault("fusion_interval_hz", 4.0)
    health = state.setdefault("health", {})
    health["fresh"] = True
    health["age_ms"] = 0
    health.setdefault("baseline_locked", False)
    existing = state.setdefault("sensors", {})
    existing.update(sensors)  # declared channels; soma_motor_report untouched
    state["engineered_signal"] = True  # provenance marker in the file itself
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--wave", action="store_true", help="declared waveform (default)")
    mode.add_argument("--constant", type=float, default=None,
                      help="pin every channel to V — negative control")
    mode.add_argument("--drift", action="store_true",
                      help="presence ramp 0.2->0.9 over --duration")
    ap.add_argument("--state-path",
                    default=os.environ.get("SOMA_STATE_PATH", "/tmp/soma_state.json"))
    ap.add_argument("--interval", type=float, default=0.25,
                    help="seconds between writes (default 0.25 — faster than "
                         "the sensor sampler so the signal isn't aliased flat)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="stop after N seconds (0 = run until killed)")
    ap.add_argument("--once", action="store_true", help="single write then exit")
    args = ap.parse_args()

    t0 = time.time()
    while True:
        t = time.time() - t0
        if args.constant is not None:
            vals = constant_values(args.constant)
        elif args.drift:
            vals = drift_values(t, args.duration or 900.0)
        else:
            vals = wave_values(t)
        write_state(args.state_path, vals)
        if args.once:
            break
        if args.duration and t >= args.duration:
            break
        time.sleep(args.interval)
        print(f"t={t:6.1f}s presence={vals['soma_presence']:.3f} "
              f"schumann={vals['soma_schumann']:.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
