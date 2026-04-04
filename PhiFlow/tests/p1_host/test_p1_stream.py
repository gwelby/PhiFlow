from __future__ import annotations

from p1_host.host import P1Host


def _demo_wat() -> str:
    # Identical fixture to tests/p1_host/test_p1_host.py
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


def test_stream_produces_n_snapshots() -> None:
    host = P1Host()
    snapshots = list(host.stream(_demo_wat(), cycles=3))

    assert len(snapshots) == 3
    for snap in snapshots:
        assert 0.0 < snap.final_coherence < 1.0
        assert len(snap.sensor_readings) >= 1
        assert snap.sensor_readings[0].cpu_percent != 47.0
        assert snap.sensor_readings[0].cpu_percent != 0.618
        assert isinstance(snap.wasm_return_value, float)


def test_stream_resets_state_between_cycles() -> None:
    host = P1Host()
    snapshots = list(host.stream(_demo_wat(), cycles=2))

    for snap in snapshots:
        assert snap.intention_stack_final == []
        assert len(snap.resonance_log) >= 1


def test_prior_coherence_parameter_accepted() -> None:
    host = P1Host()
    snap = host.run(_demo_wat(), prior_coherence=0.0)
    assert 0.0 < snap.final_coherence < 1.0

    snap_high = host.run(_demo_wat(), prior_coherence=1.0)
    assert 0.0 < snap_high.final_coherence < 1.0
    assert isinstance(snap_high.final_coherence, float)
