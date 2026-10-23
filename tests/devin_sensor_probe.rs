// Devin diagnostic — is the live sampler populating host sensors?
// Run: cargo test --test devin_sensor_probe -- --nocapture
use phiflow::phi_ir::SensorKind;
use phiflow::sensors::read_sensor;
use std::time::{Duration, Instant};

#[test]
fn sampler_populates_memory_within_2s() {
    let t0 = Instant::now();
    std::thread::sleep(Duration::from_secs(2));
    let mem = read_sensor(SensorKind::MemoryUsage);
    let cpu = read_sensor(SensorKind::CpuUsage);
    eprintln!("after {:?}: memory_usage={:?} cpu_usage={:?}", t0.elapsed(), mem, cpu);
    assert!(mem.is_some(), "memory_usage stayed None — sampler not populating");
}

#[test]
fn sampler_thread_spawns_on_first_read() {
    let before = std::fs::read_to_string("/proc/self/status")
        .map(|s| s.lines().find(|l| l.starts_with("Threads")).unwrap_or("").to_string())
        .unwrap_or_default();
    let _ = read_sensor(SensorKind::MemoryUsage);
    std::thread::sleep(Duration::from_millis(500));
    let after = std::fs::read_to_string("/proc/self/status")
        .map(|s| s.lines().find(|l| l.starts_with("Threads")).unwrap_or("").to_string())
        .unwrap_or_default();
    eprintln!("{} -> {}", before, after);
}

/// Ghost-path test: StreamPush carries Option<f64> threshold and the evaluator
/// enforces it (force-yield when coherence drops below), but lowering.rs:583
/// always emits None — no .phi program can declare one. This test lifts the
/// council program out of a real daemon state snapshot, patches the threshold
/// to Some(0.55), and checks whether the yield path actually fires.
/// Run: cargo test --release --test devin_sensor_probe threshold -- --nocapture --ignored
#[test]
#[ignore = "requires /tmp/phiflow_daemon_state.json from a live daemon run"]
fn stream_threshold_yield_is_reachable_via_ir() {
    use phiflow::phi_ir::evaluator::{Evaluator, VmExecResult};
    use phiflow::phi_ir::{PhiIRNode, PhiIRProgram};

    let raw = std::fs::read_to_string("/tmp/phiflow_daemon_state.json")
        .expect("daemon state file missing — run phic --daemon once first");
    let state: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let prog_json = state["council"]["program"].clone();
    let mut prog: PhiIRProgram = serde_json::from_value(prog_json).unwrap();

    let mut patched = false;
    for block in &mut prog.blocks {
        for instr in &mut block.instructions {
            if let PhiIRNode::StreamPush(name, t) = &mut instr.node {
                if name == "await_substrate" {
                    *t = Some(0.55);
                    patched = true;
                }
            }
        }
    }
    assert!(patched, "await_substrate StreamPush not found in council IR");

    let mut eval = Evaluator::new(prog);
    eval.max_steps = Some(50_000);
    match eval.run_or_yield() {
        Ok(VmExecResult::Yielded { snapshot, .. }) => {
            eprintln!("YIELDED at coherence {:.4} — threshold path WORKS", snapshot.coherence);
            assert!(snapshot.coherence < 0.55 + 0.001);
        }
        other => panic!("expected Yielded below threshold, got: {:?}", other),
    }
}
