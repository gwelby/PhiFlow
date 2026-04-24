use std::fs;
use std::sync::Arc;
use phiflow::host::PhiHostProvider;
use phiflow::system_host::SystemHostProvider;
use phiflow::security::anchor::AnchorSigningKey;

fn setup_soma_mock(dir: &std::path::Path) {
    let soma_path = dir.join("soma_state.json");
    let ts = chrono::Utc::now().to_rfc3339();
    let soma_json = serde_json::json!({
        "schema_version": "soma.phiflow.v1",
        "updated_at": ts,
        "session_id": "test-session",
        "runtime": {
            "sensor_stack": "mock",
            "ring_sensor_type": "mock",
            "sample_rate_hz": 1.0,
            "fusion_interval_hz": 1.0
        },
        "health": {
            "fresh": true,
            "age_ms": 0,
            "baseline_locked": true
        },
        "sensors": {
            "soma_schumann": 7.83,
            "soma_432": 432.0,
            "soma_presence": 1.0,
            "soma_fan_hz": 1.0,
            "soma_ac_60": 60.0,
            "soma_peak_dbc": 1.0,
            "ring_slope_1f": 1.0,
            "ring_jitter_ns": 1.0,
            "ring_coherence_432": 1.0,
            "ring_coherence_528": 1.0,
            "ring_phase_delta": 1.0
        }
    });
    fs::write(&soma_path, soma_json.to_string()).expect("Failed to write mock SOMA");
    std::env::set_var("SOMA_STATE_PATH", soma_path.to_str().unwrap());
}

#[test]
fn test_system_host_signed_handoff() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let base_path = temp_dir.path().to_path_buf();
    setup_soma_mock(&base_path);
    
    let key = Arc::new(AnchorSigningKey::generate());
    let host = SystemHostProvider::new(base_path.clone(), key.clone());
    
    // Broadcast a mock handoff
    let payload = serde_json::json!({
        "target": "lumi",
        "task_id": "T-123",
        "context": "test context"
    });
    
    host.broadcast("_handoff", &payload.to_string());
    
    // Check if the handoff file was created
    let handoff_file = base_path.join("channel__handoff.jsonl");
    assert!(handoff_file.exists(), "Handoff file should be created");
    
    let content = fs::read_to_string(handoff_file).expect("Failed to read handoff file");
    let line = content.lines().next().expect("Expected at least one line");
    
    let parsed: serde_json::Value = serde_json::from_str(line).expect("Must be valid JSON");
    
    // Verify it's an attestation (has signatures and fingerprints)
    assert!(parsed.get("signature").is_some(), "Should contain a signature");
    assert_ne!(parsed["signature"], "unsigned", "Signature should not be 'unsigned'");
    assert_eq!(parsed["key_fingerprint"], key.fingerprint(), "Fingerprint should match");

    assert!(parsed.get("signature_pq").is_some(), "Should contain a PQ signature");
    assert_ne!(parsed["signature_pq"], "none", "PQ Signature should not be 'none'");
    assert_eq!(parsed["key_fingerprint_pq"], key.fingerprint_pq(), "PQ Fingerprint should match");
}

#[test]
fn test_system_host_ledger_requires_system_intent() {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let base_path = temp_dir.path().to_path_buf();
    setup_soma_mock(&base_path);
    
    let key = Arc::new(AnchorSigningKey::generate());
    let host = SystemHostProvider::new(base_path.clone(), key.clone());
    
    // 1. Try to broadcast to ledger WITHOUT system intent
    host.broadcast("ledger", "{\"context\":\"unauthorized\"}");
    
    // Since SYSTEM intent is not active, it should fall back to base_path/channel_ledger.jsonl
    let fake_ledger = base_path.join("channel_ledger.jsonl");
    assert!(fake_ledger.exists(), "Should write to fake ledger when non-system");
}
