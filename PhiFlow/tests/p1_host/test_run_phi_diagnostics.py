from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import run_phi


def _valid_diag() -> dict[str, object]:
    return {
        "error_code": "E001_UNEXPECTED_TOKEN",
        "line": 3,
        "column": 0,
        "found": "State",
        "expected": None,
        "hint": "This keyword is not valid at the start of an expression.",
        "example_fix": "let value = some_function()\nresonate value",
    }


def test_classify_success_exit_zero() -> None:
    status, diagnostics = run_phi._classify_phic_json_errors(0, "", "")
    assert status == 0
    assert diagnostics is None


def test_classify_parse_failure_exit_two_with_strict_schema() -> None:
    payload = json.dumps([_valid_diag()])
    status, diagnostics = run_phi._classify_phic_json_errors(2, payload, "")
    assert status == 2
    assert diagnostics is not None
    assert diagnostics[0]["error_code"] == "E001_UNEXPECTED_TOKEN"
    assert set(diagnostics[0].keys()) == run_phi.REQUIRED_DIAGNOSTIC_FIELDS


def test_classify_io_runtime_failure_exit_one() -> None:
    status, diagnostics = run_phi._classify_phic_json_errors(1, "", "io failure")
    assert status == 1
    assert diagnostics is None


def test_schema_rejects_additional_fields() -> None:
    bad = _valid_diag()
    bad["extra"] = "not allowed"
    with pytest.raises(ValueError, match="schema mismatch"):
        run_phi._parse_json_diagnostics(json.dumps([bad]))


def test_schema_rejects_missing_fields() -> None:
    bad = _valid_diag()
    del bad["hint"]
    with pytest.raises(ValueError, match="schema mismatch"):
        run_phi._parse_json_diagnostics(json.dumps([bad]))


def test_schema_rejects_non_array_top_level() -> None:
    with pytest.raises(ValueError, match="must be a JSON array"):
        run_phi._parse_json_diagnostics(json.dumps(_valid_diag()))


def test_schema_rejects_mixed_prose_and_json() -> None:
    mixed = "Parse Error:\n" + json.dumps([_valid_diag()])
    with pytest.raises(ValueError, match="not valid JSON"):
        run_phi._parse_json_diagnostics(mixed)


def test_run_phic_json_errors_uses_subprocess_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_binary = tmp_path / "phic"
    fake_phi = tmp_path / "program.phi"
    fake_phi.write_text("intention \"x\" {", encoding="utf-8")

    monkeypatch.setattr(run_phi, "_resolve_phic_binary", lambda _repo_root: fake_binary)

    def fake_run(cmd: list[str], capture_output: bool, text: bool, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert cmd == [str(fake_binary), str(fake_phi), "--json-errors"]
        assert capture_output is True
        assert text is True
        assert cwd == tmp_path
        return subprocess.CompletedProcess(cmd, 2, stdout=json.dumps([_valid_diag()]), stderr="")

    monkeypatch.setattr(run_phi.subprocess, "run", fake_run)

    status, diagnostics = run_phi._run_phic_json_errors(fake_phi, tmp_path)
    assert status == 2
    assert diagnostics is not None
    assert diagnostics[0]["error_code"] == "E001_UNEXPECTED_TOKEN"
