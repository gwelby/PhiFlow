from __future__ import annotations

from pathlib import Path

from p1_host.host import P1Host


def _precompiled_demo_wat() -> str:
    # Precompiled fixture equivalent to examples/p1_demo.phi semantics.
    return """
(module
  (import "phi" "witness" (func $phi_witness (param i32) (result f64)))
  (import "phi" "resonate" (func $phi_resonate (param f64)))
  (import "phi" "coherence" (func $phi_coherence (result f64)))
  (import "phi" "intention_push" (func $phi_intention_push (param i32)))
  (import "phi" "intention_pop" (func $phi_intention_pop))

  (memory (export "memory") 1)
  (data (i32.const 256) "healing_session")
  (global $string_len (export "string_len") (mut i32) (i32.const 0))

  (func (export "phi_run") (result f64)
    i32.const 15
    global.set $string_len
    i32.const 256
    call $phi_intention_push

    i32.const 0
    call $phi_witness
    drop

    f64.const 432.0
    call $phi_resonate

    call $phi_coherence
    drop

    call $phi_intention_pop

    call $phi_coherence
  )
)
"""


def test_host_runs_demo_program() -> None:
    # Ensure source-level demo exists in repository.
    demo_path = Path("examples/p1_demo.phi")
    assert demo_path.exists(), "examples/p1_demo.phi must exist"

    host = P1Host()
    snapshot = host.run(_precompiled_demo_wat())

    assert 0.0 < snapshot.final_coherence < 1.0
    assert len(snapshot.sensor_readings) >= 1
    assert snapshot.sensor_readings[0].cpu_percent is not None
    assert snapshot.sensor_readings[0].cpu_percent != 47.0
    assert snapshot.sensor_readings[0].cpu_percent != 0.618
    assert isinstance(snapshot.wasm_return_value, float)

    # All hooks were exercised by the fixture.
    assert len(snapshot.resonance_log) >= 1
    assert snapshot.intention_stack_final == []