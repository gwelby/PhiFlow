//! SOMA coherence integration tests.
//!
//! Verifies that live SOMA sensor data influences the coherence value
//! returned by `compute_coherence_from_sensors()`. This is the test that
//! proves coherence is a *measurement*, not just a formula — when SOMA
//! reports low environmental presence, the coherence value should drop.
//!
//! These tests MUST run single-threaded because they share a process-wide
//! OnceLock for the sensor data thread. Run with:
//!   cargo test --test soma_coherence_test -- --test-threads=1

use phiflow::sensors::{compute_coherence_from_sensors, is_soma_state_fresh, SomaState};
use std::fs;

fn write_soma_state(path: &str, presence: f64, fan_hz: f64, peak_dbc: f64) {
    let content = format!(
        r#"{{
  "schema_version": "soma.phiflow.v1",
  "updated_at": "2099-01-01T00:00:00",
  "session_id": "test",
  "runtime": {{
    "sensor_stack": "SOMA",
    "ring_sensor_type": "CPU",
    "sample_rate_hz": 500.0,
    "fusion_interval_hz": 5.0
  }},
  "health": {{
    "fresh": true,
    "age_ms": 0,
    "baseline_locked": true
  }},
  "sensors": {{
    "soma_schumann": 0.0,
    "soma_432": 0.0,
    "soma_presence": {},
    "soma_fan_hz": {},
    "soma_ac_60": 0.0,
    "soma_peak_dbc": {},
    "ring_slope_1f": 0.0,
    "ring_jitter_ns": 0.0,
    "ring_coherence_432": 0.0,
    "ring_coherence_528": 0.0,
    "ring_phase_delta": 0.0
  }}
}}"#,
        presence, fan_hz, peak_dbc
    );
    fs::write(path, content).unwrap();
}

#[test]
fn test_soma_state_parses_and_is_fresh() {
    let tmp = std::env::temp_dir().join("soma_test_parse.json");
    write_soma_state(tmp.to_str().unwrap(), 0.5, 50.0, 15.0);
    let content = fs::read_to_string(&tmp).unwrap();
    let state: SomaState = serde_json::from_str(&content).unwrap();
    assert_eq!(state.schema_version, "soma.phiflow.v1");
    assert!(state.health.fresh);
    assert!(is_soma_state_fresh(&state));
    assert!((state.sensors.soma_presence - 0.5).abs() < 1e-10);
}

#[test]
fn test_coherence_without_soma_is_in_valid_range() {
    // Point SOMA_STATE_PATH to a non-existent file.
    std::env::set_var("SOMA_STATE_PATH", "/tmp/nonexistent_soma_test_12345.json");
    let coherence = compute_coherence_from_sensors();
    assert!(
        coherence >= 0.0 && coherence <= 1.0,
        "Coherence should be in [0, 1], got {}",
        coherence
    );
}

#[test]
fn test_coherence_with_high_presence_is_higher_than_low_presence() {
    // Write a SOMA state with high presence (stable environment)
    let tmp_high = std::env::temp_dir().join("soma_test_high_presence.json");
    write_soma_state(tmp_high.to_str().unwrap(), 0.95, 50.0, 25.0);
    std::env::set_var("SOMA_STATE_PATH", tmp_high.to_str().unwrap());

    // Wait for the sensor thread to pick up the new file.
    // The thread polls every 100ms.
    std::thread::sleep(std::time::Duration::from_millis(300));
    let coherence_high = compute_coherence_from_sensors();

    // Write a SOMA state with low presence (noisy environment)
    let tmp_low = std::env::temp_dir().join("soma_test_low_presence.json");
    write_soma_state(tmp_low.to_str().unwrap(), 0.05, 50.0, 5.0);
    std::env::set_var("SOMA_STATE_PATH", tmp_low.to_str().unwrap());

    std::thread::sleep(std::time::Duration::from_millis(300));
    let coherence_low = compute_coherence_from_sensors();

    // The high-presence coherence should be higher than low-presence.
    // This proves SOMA data is flowing into the coherence calculation.
    assert!(
        coherence_high > coherence_low,
        "High presence (0.95) should produce higher coherence ({}) than low presence (0.05) ({})",
        coherence_high,
        coherence_low
    );

    // Also verify the low-presence value is noticeably degraded
    assert!(
        coherence_low < 0.9,
        "Low presence (0.05) should degrade coherence below 0.9, got {}",
        coherence_low
    );
}

#[test]
fn test_coherence_drops_with_degraded_soma() {
    // Write a SOMA state with very low presence (degraded environment)
    let tmp = std::env::temp_dir().join("soma_test_degraded_presence.json");
    write_soma_state(tmp.to_str().unwrap(), 0.01, 0.0, 1.0);
    std::env::set_var("SOMA_STATE_PATH", tmp.to_str().unwrap());

    // Wait for the sensor thread to pick up the file.
    std::thread::sleep(std::time::Duration::from_millis(500));
    let degraded = compute_coherence_from_sensors();

    // With presence=0.01, fan=0, peak_dbc=1.0, the coherence should be
    // well below the pure-formula value. The system coherence is typically
    // ~1.0 on an idle machine, so the SOMA blend should pull it down to
    // roughly (1.0 + 0.01*0.3 + 0.033*0.1) / 1.4 ≈ 0.72.
    assert!(
        degraded < 0.85,
        "Degraded SOMA (presence=0.01, fan=0, peak_dbc=1.0) should pull coherence \
         below 0.85, got {}",
        degraded
    );
}
