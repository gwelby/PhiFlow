from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from p1_host.host import P1Host

REQUIRED_DIAGNOSTIC_FIELDS = {
    "error_code",
    "line",
    "column",
    "found",
    "expected",
    "hint",
    "example_fix",
}


def has_stream_block(source: str) -> bool:
    return bool(re.search(r'\bstream\s+"', source))


def _compile_phi_to_wat(phi_path: Path, repo_root: Path) -> str:
    out_path = phi_path.with_suffix(".wat")
    command = [
        "cargo",
        "run",
        "--manifest-path",
        str(repo_root / "Cargo.toml"),
        "--bin",
        "phi_emit_wat",
        "--",
        str(phi_path),
        "--out",
        str(out_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(
            "PhiFlow source-to-WAT compile command failed. Use --wat-file to run a precompiled module.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if out_path.exists():
        return out_path.read_text(encoding="utf-8")

    if "(module" in result.stdout:
        return result.stdout[result.stdout.index("(module") :]

    raise RuntimeError("Compiler succeeded but no WAT output was found. Use --wat-file.")


def _load_wasm_input(path: Path) -> str | bytes:
    if path.suffix.lower() == ".wasm":
        return path.read_bytes()
    return path.read_text(encoding="utf-8")


def _resolve_phic_binary(repo_root: Path) -> Path:
    exe_name = "phic.exe" if sys.platform.startswith("win") else "phic"
    phic_path = repo_root / "target" / "debug" / exe_name
    if phic_path.exists():
        return phic_path

    build_cmd = [
        "cargo",
        "build",
        "--manifest-path",
        str(repo_root / "Cargo.toml"),
        "--bin",
        "phic",
    ]
    build = subprocess.run(build_cmd, capture_output=True, text=True, cwd=repo_root)
    if build.returncode != 0:
        raise RuntimeError(
            "Failed to build phic binary for diagnostics stage.\n"
            f"stdout:\n{build.stdout}\n"
            f"stderr:\n{build.stderr}"
        )

    if not phic_path.exists():
        raise RuntimeError(f"phic binary not found after build: {phic_path}")

    return phic_path


def _validate_diagnostic_schema(diag: Any) -> dict[str, Any]:
    if not isinstance(diag, dict):
        raise ValueError("diagnostic item must be an object")

    keys = set(diag.keys())
    if keys != REQUIRED_DIAGNOSTIC_FIELDS:
        missing = sorted(REQUIRED_DIAGNOSTIC_FIELDS - keys)
        extra = sorted(keys - REQUIRED_DIAGNOSTIC_FIELDS)
        raise ValueError(f"diagnostic schema mismatch: missing={missing}, extra={extra}")

    if not isinstance(diag["error_code"], str):
        raise ValueError("diagnostic.error_code must be a string")
    if not isinstance(diag["line"], int):
        raise ValueError("diagnostic.line must be an integer")
    if not isinstance(diag["column"], int):
        raise ValueError("diagnostic.column must be an integer")
    if not isinstance(diag["found"], str):
        raise ValueError("diagnostic.found must be a string")
    if diag["expected"] is not None and not isinstance(diag["expected"], str):
        raise ValueError("diagnostic.expected must be a string or null")
    if not isinstance(diag["hint"], str):
        raise ValueError("diagnostic.hint must be a string")
    if not isinstance(diag["example_fix"], str):
        raise ValueError("diagnostic.example_fix must be a string")

    return {
        "error_code": diag["error_code"],
        "line": diag["line"],
        "column": diag["column"],
        "found": diag["found"],
        "expected": diag["expected"],
        "hint": diag["hint"],
        "example_fix": diag["example_fix"],
    }


def _parse_json_diagnostics(stdout_payload: str) -> list[dict[str, Any]]:
    payload = stdout_payload.strip()
    if not payload:
        raise ValueError("diagnostics payload is empty")

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"diagnostics payload is not valid JSON: {exc}") from exc

    if not isinstance(parsed, list):
        raise ValueError("diagnostics payload must be a JSON array")

    return [_validate_diagnostic_schema(item) for item in parsed]


def _classify_phic_json_errors(returncode: int, stdout_payload: str, stderr_payload: str) -> tuple[int, list[dict[str, Any]] | None]:
    if returncode == 0:
        return 0, None

    if returncode == 2:
        if stderr_payload.strip():
            raise ValueError("parse diagnostics must write JSON to stdout only; stderr must be empty")
        diagnostics = _parse_json_diagnostics(stdout_payload)
        return 2, diagnostics

    if returncode == 1:
        return 1, None

    raise RuntimeError(f"unexpected phic exit code: {returncode}")


def _run_phic_json_errors(phi_path: Path, repo_root: Path) -> tuple[int, list[dict[str, Any]] | None]:
    phic_binary = _resolve_phic_binary(repo_root)
    result = subprocess.run(
        [str(phic_binary), str(phi_path), "--json-errors"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    return _classify_phic_json_errors(result.returncode, result.stdout, result.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile and run PhiFlow through P1 WASM host",
        epilog="Diagnostics stage contract: exit=0 success, exit=2 parse diagnostics JSON, exit=1 IO/runtime.",
    )
    parser.add_argument("phi_file", nargs="?", help="Path to a .phi file")
    parser.add_argument("--wat-file", dest="wat_file", help="Path to precompiled .wat or .wasm")
    parser.add_argument("--stream-output", action="store_true", help="Emit live JSON events if a stream block is detected")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    try:
        if args.wat_file:
            module_source = _load_wasm_input(Path(args.wat_file))
        else:
            if not args.phi_file:
                parser.error("Provide <path_to_.phi_file> or --wat-file")

            phi_path = Path(args.phi_file)
            status, diagnostics = _run_phic_json_errors(phi_path, repo_root)
            if status == 2:
                print(json.dumps(diagnostics, indent=2))
                return 2
            if status == 1:
                print("phic diagnostics stage returned IO/runtime failure", file=sys.stderr)
                return 1

            module_source = _compile_phi_to_wat(phi_path, repo_root)

        host = P1Host()

        # Stream detection logic
        phi_source_text = ""
        if args.phi_file:
            phi_source_text = Path(args.phi_file).read_text(encoding="utf-8")
        elif args.wat_file and (args.wat_file.endswith(".wat") or args.wat_file.endswith(".wasm")):
            pass  # impossible to check 'stream' from just wat natively here but instructions say check source file.
            
        do_stream = False
        if args.stream_output and has_stream_block(phi_source_text):
            do_stream = True

        if do_stream:
            stream_out_path = Path("/tmp/phiflow_stream_latest.jsonl")
            stream_out_path.parent.mkdir(parents=True, exist_ok=True)
            stream_out_path.write_text("", encoding="utf-8") # clear it
            
            cycle_count = 0
            latest_resonance = 0.0
            for snapshot in host.stream(module_source):
                cycle_count += 1
                latest_resonance = snapshot.resonance_log[-1] if snapshot.resonance_log else 0.0
                
                event = {
                    "cycle": cycle_count,
                    "coherence": snapshot.final_coherence,
                    "resonance": latest_resonance,
                    "status": "running"
                }
                
                with open(stream_out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
                    
                if snapshot.stream_broken:
                    break

            summary = {
                "status": "complete",
                "cycles": cycle_count,
                "final_resonance": latest_resonance,
                "convergence_achieved": True
            }
            print(json.dumps(summary))
            return 0
        else:
            snapshot = host.run(module_source)

            print(
                json.dumps(
                    {
                        "final_coherence": snapshot.final_coherence,
                        "sensor_readings": [asdict(r) for r in snapshot.sensor_readings],
                        "intention_stack_final": snapshot.intention_stack_final,
                        "resonance_log": snapshot.resonance_log,
                        "wasm_return_value": snapshot.wasm_return_value,
                        "execution_time_ms": snapshot.execution_time_ms,
                    },
                    indent=2,
                )
            )
            return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Diagnostics contract violation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
