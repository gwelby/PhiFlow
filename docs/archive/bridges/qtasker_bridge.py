"""
PhiFlow → QTasker Autonomous Bridge
====================================
Monitors PhiFlow's handoff channel log and forwards autonomous action events
to the QTasker P1 Autonomous Gateway on 127.0.0.1:18889.

How it works:
  1. Tails /mnt/d/Projects/PhiFlow/channel__handoff.jsonl (or the configured
     PHIFLOW_HANDOFF_LOG env var) from the current end of file.
  2. For each new NDJSON line that can be parsed, extracts intention/coherence
     fields and POSTs a handoff payload to QTasker.
  3. Retries with exponential back-off if QTasker is temporarily unreachable.
  4. Also watches soma_state.json for high-coherence spikes and forwards those
     as autonomous "coherence_peak" action events (if PHIFLOW_SOMA_WATCH=1).

Usage:
  python bridges/qtasker_bridge.py                     # background tailing
  PHIFLOW_SOMA_WATCH=1 python bridges/qtasker_bridge.py

Environment variables:
  PHIFLOW_HANDOFF_LOG  — path to handoff channel log
                         (default: ./channel__handoff.jsonl)
  QTASKER_P1_URL       — full URL of the P1 gateway endpoint
                         (default: http://127.0.0.1:18889/handoff)
  QTASKER_P1_TOKEN     — shared-secret Bearer token (if QTasker requires one)
  PHIFLOW_SOMA_WATCH   — set to "1" to also watch soma_state.json
  SOMA_STATE_PATH      — path to soma_state.json (default: SOMA canonical path)
  COHERENCE_THRESHOLD  — minimum coherence value to trigger a soma event (0.0-1.0,
                         default 0.85)
  POLL_INTERVAL_S      — seconds between tail reads (default 0.25)
  MAX_RETRY            — max retry attempts per failed send (default 5)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

# ── Config ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

HANDOFF_LOG = Path(
    os.environ.get(
        "PHIFLOW_HANDOFF_LOG",
        str(REPO_ROOT / "channel__handoff.jsonl"),
    )
)
QTASKER_URL = os.environ.get("QTASKER_P1_URL", "http://127.0.0.1:18889/handoff")
P1_TOKEN = os.environ.get("QTASKER_P1_TOKEN", "")
SOMA_STATE_PATH = Path(
    os.environ.get(
        "SOMA_STATE_PATH",
        "/mnt/d/Projects/PhiHarmonic/SOMA/soma_state.json",
    )
)
SOMA_WATCH = os.environ.get("PHIFLOW_SOMA_WATCH", "0") == "1"
COHERENCE_THRESHOLD = float(os.environ.get("COHERENCE_THRESHOLD", "0.85"))
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL_S", "0.25"))
MAX_RETRY = int(os.environ.get("MAX_RETRY", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [QT-BRIDGE] %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("qtasker_bridge")


# ── HTTP sender ───────────────────────────────────────────────────────────────
def _send_handoff(payload: Dict[str, Any]) -> bool:
    """POST payload to QTasker. Returns True on success."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        QTASKER_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
    )
    if P1_TOKEN:
        req.add_header("Authorization", f"Bearer {P1_TOKEN}")

    delay = 1.0
    for attempt in range(1, MAX_RETRY + 1):
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                log.info("✅ Task created: %s (id=%s)", result.get("name"), result.get("task_id"))
                return True
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            log.warning("HTTP %d from QTasker (attempt %d/%d): %s", exc.code, attempt, MAX_RETRY, body_text)
            if exc.code in (400, 401, 404):
                return False  # Not retryable
        except (urllib.error.URLError, OSError) as exc:
            log.warning("Connection error (attempt %d/%d): %s", attempt, MAX_RETRY, exc)

        if attempt < MAX_RETRY:
            log.info("Retrying in %.1fs…", delay)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)

    log.error("❌ Failed to deliver handoff after %d attempts", MAX_RETRY)
    return False


# ── Handoff log parser ────────────────────────────────────────────────────────
def _parse_handoff_line(raw: str) -> Dict[str, Any] | None:
    """Parse an NDJSON handoff line into a QTasker payload, or None if irrelevant."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(record, dict):
        return None

    # Signed envelope wraps the actual payload under "payload" key
    inner = record
    if "payload" in record and isinstance(record["payload"], (str, dict)):
        if isinstance(record["payload"], str):
            try:
                inner = json.loads(record["payload"])
            except json.JSONDecodeError:
                inner = record
        else:
            inner = record["payload"]

    action = inner.get("action") or inner.get("context") or inner.get("report") or ""
    if not action:
        # Try common PhiFlow event fields
        action = inner.get("type") or inner.get("event") or inner.get("message") or ""
    if not action:
        return None  # Nothing actionable

    return {
        "intention": inner.get("intention") or inner.get("agent") or "phiflow",
        "coherence": float(inner.get("coherence") or inner.get("soma_presence") or 0.0),
        "action": str(action),
        "context": str(inner.get("context") or inner.get("report") or ""),
        "priority": str(inner.get("priority") or "medium"),
        "tags": list(inner.get("tags") or []),
        "source": "phiflow-handoff",
    }


# ── Handoff log tailer ────────────────────────────────────────────────────────
def tail_handoff_log() -> None:
    """Tail PHIFLOW_HANDOFF_LOG and forward new events to QTasker."""
    log.info("📂 Watching handoff log: %s", HANDOFF_LOG)
    # Create the file if it doesn't exist yet (PhiFlow may not have run)
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_LOG.touch(exist_ok=True)

    with open(HANDOFF_LOG, "r", encoding="utf-8") as fh:
        fh.seek(0, os.SEEK_END)  # Start from current end
        while True:
            line = fh.readline()
            if not line:
                time.sleep(POLL_INTERVAL)
                continue
            payload = _parse_handoff_line(line)
            if payload:
                log.info("🔁 Forwarding: intention=%s action=%s", payload["intention"], payload["action"])
                _send_handoff(payload)


# ── SOMA watcher ──────────────────────────────────────────────────────────────
def watch_soma_state() -> None:
    """Watch soma_state.json for coherence spikes and forward them as actions."""
    log.info("🧠 Watching SOMA state: %s (threshold=%.2f)", SOMA_STATE_PATH, COHERENCE_THRESHOLD)
    last_ts: str | None = None
    last_coherence: float = 0.0

    while True:
        time.sleep(POLL_INTERVAL * 4)  # Slower poll; SOMA updates ~1Hz
        if not SOMA_STATE_PATH.exists():
            continue
        try:
            state = json.loads(SOMA_STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        ts = state.get("updated_at") or state.get("timestamp") or ""
        coherence = float(state.get("coherence") or state.get("overall_coherence") or 0.0)

        # Only fire if timestamp changed and coherence crossed the threshold
        if ts == last_ts:
            continue
        last_ts = ts

        if coherence >= COHERENCE_THRESHOLD and abs(coherence - last_coherence) > 0.05:
            last_coherence = coherence
            payload = {
                "intention": state.get("profile") or "soma-peak",
                "coherence": coherence,
                "action": f"SOMA coherence peak {coherence:.3f}",
                "context": (
                    f"low_band={state.get('low_band_activity')}, "
                    f"bins={list(state.get('harmonic_bins', {}).keys())[:4]}"
                ),
                "priority": "high" if coherence >= 0.95 else "medium",
                "tags": ["soma", "coherence-peak"],
                "source": "soma-watcher",
            }
            log.info("⚡ SOMA peak %.3f — forwarding to QTasker", coherence)
            _send_handoff(payload)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading

    log.info("🌉 PhiFlow → QTasker Bridge starting")
    log.info("   Handoff log : %s", HANDOFF_LOG)
    log.info("   QTasker URL : %s", QTASKER_URL)
    log.info("   SOMA watch  : %s", SOMA_WATCH)

    threads = [threading.Thread(target=tail_handoff_log, daemon=True, name="handoff-tailer")]
    if SOMA_WATCH:
        threads.append(threading.Thread(target=watch_soma_state, daemon=True, name="soma-watcher"))

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log.info("Bridge dissolving…")
        sys.exit(0)
