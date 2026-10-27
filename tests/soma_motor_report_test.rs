//! SomaMotorReport sensor regression tests (R5 engineered closure).
//!
//! The `soma_motor_report` channel is the afferent half of the R5
//! actuate→bus→report→sensor loop. It is Option<f64> by contract:
//! absent field or stale state must read None (channel absent),
//! never a phantom 0.0.
//!
//! Run single-threaded — tests share the process-wide sensor OnceLock
//! and the SOMA_STATE_PATH env var:
//!   cargo test --test soma_motor_report_test -- --test-threads=1
//!
//! Note: the sampler thread re-reads the state file every ~100ms plus
//! sysinfo sampling overhead (seconds on WSL), so tests poll read_sensor
//! until the sampler has provably re-read rather than sleeping once.

use phiflow::phi_ir::SensorKind;
use phiflow::sensors::{read_sensor, SomaState};
use std::fs;
use std::sync::Mutex;
use std::time::{Duration, Instant};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn state_path() -> std::path::PathBuf {
    std::env::temp_dir().join("soma_motor_report_test_state.json")
}

fn write_state(report: Option<&str>, fresh: bool, updated_at: &str, age_ms: u64) {
    let report_line = match report {
        Some(v) => format!(r#""soma_motor_report": {},"#, v),
        None => String::new(),
    };
    let content = format!(
        r#"{{
  "schema_version": "soma.phiflow.v1",
  "updated_at": "{}",
  "session_id": "test",
  "runtime": {{
    "sensor_stack": "SOMA",
    "ring_sensor_type": "CPU",
    "sample_rate_hz": 500.0,
    "fusion_interval_hz": 5.0
  }},
  "health": {{
    "fresh": {},
    "age_ms": {},
    "baseline_locked": true
  }},
  "sensors": {{
    {}
    "soma_presence": 0.5
  }}
}}"#,
        updated_at, fresh, age_ms, report_line
    );
    fs::write(state_path(), content).unwrap();
}

/// Poll read_sensor until it returns `expected` or the deadline passes.
/// Polling (not a fixed sleep) is what makes "the sampler re-read the
/// file" observable: a stale cached value keeps polling, the correct one
/// ends the loop.
fn wait_for(expected: Option<f64>) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        let got = read_sensor(SensorKind::SomaMotorReport);
        if got == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {:?}, last read {:?}",
            expected,
            got
        );
        std::thread::sleep(Duration::from_millis(150));
    }
}

fn guard() -> std::sync::MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn test_name_and_id_mapping() {
    assert_eq!(
        SensorKind::from_name("soma_motor_report"),
        Some(SensorKind::SomaMotorReport)
    );
    assert_eq!(SensorKind::SomaMotorReport.as_name(), "soma_motor_report");
    assert_eq!(SensorKind::SomaMotorReport.as_id(), 14);
    assert_eq!(SensorKind::from_id(14), Some(SensorKind::SomaMotorReport));
}

#[test]
fn test_fresh_report_reads_value() {
    let _guard = guard();
    std::env::set_var("SOMA_STATE_PATH", state_path());
    write_state(Some("0.42"), true, "2099-01-01T00:00:00", 0);
    wait_for(Some(0.42));
}

#[test]
fn test_absent_report_reads_none() {
    let _guard = guard();
    std::env::set_var("SOMA_STATE_PATH", state_path());
    // Prove the sampler is live on this file first — a None read while
    // soma is unloaded would be a false pass.
    write_state(Some("0.77"), true, "2099-01-01T00:00:00", 0);
    wait_for(Some(0.77));
    // Fresh, valid file with NO soma_motor_report field — old producers'
    // files must still parse, and the channel must read absent, not 0.0.
    write_state(None, true, "2099-01-01T00:00:00", 0);
    wait_for(None);
}

#[test]
fn test_stale_state_reads_none() {
    let _guard = guard();
    std::env::set_var("SOMA_STATE_PATH", state_path());
    // Prove the sampler loaded THIS file, then go stale — the resulting
    // None can only come through the freshness gate.
    write_state(Some("0.88"), true, "2099-01-01T00:00:00", 0);
    wait_for(Some(0.88));
    // Report present but the whole SOMA state is stale — freshness gate
    // must fail closed; a dead producer must not keep reporting the last
    // written value as live.
    write_state(Some("0.99"), false, "2020-01-01T00:00:00", 999_999_999);
    wait_for(None);
}

#[test]
fn test_missing_file_reads_none() {
    let _guard = guard();
    let path = state_path();
    let _ = fs::remove_file(&path);
    std::env::set_var("SOMA_STATE_PATH", &path);
    wait_for(None);
}

#[test]
fn test_backward_compat_deserialize() {
    let _guard = guard();
    // A state file produced before the field existed must still parse —
    // SomaSensors carries serde(default) for exactly this.
    write_state(None, true, "2099-01-01T00:00:00", 0);
    let content = fs::read_to_string(state_path()).unwrap();
    let state: SomaState = serde_json::from_str(&content).unwrap();
    assert_eq!(state.sensors.soma_motor_report, None);
}
