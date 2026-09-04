import json
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PHI = REPO_ROOT / "run_phi.py"

def test_stream_output_emits_per_cycle_json(tmp_path):
    phi_file = tmp_path / "test_stream.phi"
    phi_file.write_text("""
    stream "test_stream" {
        let x = 42
        resonate x
        break stream
    }
    """, encoding="utf-8")

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(REPO_ROOT / "target-antigravity")
    
    result = subprocess.run(
        [sys.executable, str(RUN_PHI), str(phi_file), "--stream-output"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env
    )

    assert result.returncode == 0, f"run_phi failed: {result.stderr}"

    # Check terminal summary
    last_line = result.stdout.strip().splitlines()[-1]
    summary = json.loads(last_line)
    assert summary.get("status") == "complete"
    assert "cycles" in summary
    assert "final_resonance" in summary

    # Check /tmp/phiflow_stream_latest.jsonl (or equivalents for windows, maybe just check if file exists if tmp is hardcoded)
    # The requirement explicitly said: Write each JSON event to `/tmp/phiflow_stream_latest.jsonl`
    stream_out = Path("/tmp/phiflow_stream_latest.jsonl")
    try:
        lines = stream_out.read_text().strip().splitlines()
        assert len(lines) >= 1
        first_event = json.loads(lines[0])
        assert first_event.get("cycle") == 1
        assert "coherence" in first_event
        assert "resonance" in first_event
        assert first_event.get("status") == "running"
    finally:
        if stream_out.exists():
            stream_out.unlink()

def test_non_stream_file_uses_existing_behavior(tmp_path):
    phi_file = tmp_path / "test_normal.phi"
    phi_file.write_text("""
    intention "test_normal" {
        resonate 0.618
    }
    """, encoding="utf-8")

    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(REPO_ROOT / "target-antigravity")

    result = subprocess.run(
        [sys.executable, str(RUN_PHI), str(phi_file), "--stream-output"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env
    )

    assert result.returncode == 0

    # Should NOT have the stream summary, should just be the normal snapshot JSON
    stdout = result.stdout.strip()
    parsed = json.loads(stdout)
    assert "final_coherence" in parsed
    assert "status" not in parsed  # stream output summary shouldn't be here
