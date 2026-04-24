/// Integration tests for the T-018/T-019 Anchor Attestation layer (Phase 2).
///
/// These tests do NOT require a live SOMA connection.
/// All observations are constructed with fake values and unique uuid v4 nonces
/// so the nonce table does not cause cross-test interference.
///
/// # Audit Notes for Lumi
///
/// Test coverage is organised into five sections:
///   A. Nonce table: register_nonce rejects duplicates within a process session
///   B. Policy enforcement: check_policy gates on presence + coherence thresholds
///   C. Hashing: deterministic SHA-256 for observation and payload
///   D. Attestation creation: unsigned and secp256k1 signed paths
///   E. JSON output + verification roundtrip

use phiflow::security::anchor::{
    attestation_to_ndjson, check_policy, create_attestation, hash_observation, hash_payload,
    register_nonce, verify_attestation, AnchorAlgorithm, AnchorError, AnchorMode,
    AnchorObservation, AnchorPolicy, AnchorSigningKey, canonical_message,
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a fake observation with a unique nonce registered in the nonce table.
fn fresh_obs(presence: f64, coherence: f64) -> AnchorObservation {
    let nonce = uuid::Uuid::new_v4().to_string();
    register_nonce(&nonce).expect("fresh uuid v4 nonce must not collide");
    AnchorObservation {
        session_id: "integration-test".to_string(),
        timestamp: "2026-04-19T21:00:00.000Z".to_string(),
        soma_presence: presence,
        ring_jitter_ns: 0.42,
        ring_slope_1f: -1.1,
        field_coherence: coherence,
        nonce,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// A. Nonce table
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn anchor_nonce_first_use_succeeds() {
    let n = format!("int-test-nonce-{}", uuid::Uuid::new_v4());
    assert!(register_nonce(&n).is_ok());
}

#[test]
fn anchor_nonce_duplicate_rejected() {
    let n = format!("int-test-dup-{}", uuid::Uuid::new_v4());
    register_nonce(&n).unwrap();
    let err = register_nonce(&n).unwrap_err();
    assert!(
        matches!(err, AnchorError::NonceReused(_)),
        "Expected NonceReused, got {:?}",
        err
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// B. Policy enforcement
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn anchor_policy_passes_high_presence() {
    let obs = fresh_obs(0.95, 0.8);
    assert!(check_policy(&obs, &AnchorPolicy::attest()).is_ok());
}

#[test]
fn anchor_policy_rejects_zero_presence() {
    let obs = fresh_obs(0.0, 0.8);
    let err = check_policy(&obs, &AnchorPolicy::attest()).unwrap_err();
    assert!(matches!(err, AnchorError::PolicyViolation(_)));
    if let AnchorError::PolicyViolation(msg) = err {
        assert!(msg.contains("soma_presence"));
    }
}

#[test]
fn anchor_policy_rejects_low_coherence_when_threshold_set() {
    let obs = fresh_obs(0.9, 0.2);
    let policy = AnchorPolicy {
        min_field_coherence: 0.5,
        mode: AnchorMode::Attest,
        ..Default::default()
    };
    let err = check_policy(&obs, &policy).unwrap_err();
    assert!(matches!(err, AnchorError::PolicyViolation(_)));
    if let AnchorError::PolicyViolation(msg) = err {
        assert!(msg.contains("field_coherence"));
    }
}

#[test]
fn anchor_policy_at_exact_threshold_boundary_passes() {
    // min_presence = 0.3: exactly 0.3 must pass (>= not >)
    let obs = fresh_obs(0.3, 0.8);
    assert!(check_policy(&obs, &AnchorPolicy::attest()).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// C. Hashing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn anchor_hash_observation_is_deterministic() {
    let obs = fresh_obs(0.9, 0.8);
    let h1 = hash_observation(&obs).unwrap();
    let h2 = hash_observation(&obs).unwrap();
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64);
}

#[test]
fn anchor_hash_changes_with_different_presence_values() {
    let obs_hi = fresh_obs(0.9, 0.8);
    let obs_lo = fresh_obs(0.1, 0.8);
    assert_ne!(
        hash_observation(&obs_hi).unwrap(),
        hash_observation(&obs_lo).unwrap()
    );
}

#[test]
fn anchor_hash_payload_produces_64_char_lowercase_hex() {
    let h = hash_payload(b"PhiFlow-Attestation-v1 test vector");
    assert_eq!(h.len(), 64);
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn anchor_canonical_message_contains_all_three_hashes() {
    let msg = canonical_message("aabbcc", "ddeeff", "1.0.0");
    assert!(msg.starts_with("PhiFlow-Attestation-v1\n"));
    assert!(msg.contains("payload_hash=aabbcc"));
    assert!(msg.contains("observation_hash=ddeeff"));
    assert!(msg.contains("policy_version=1.0.0"));
}

// ─────────────────────────────────────────────────────────────────────────────
// D. Attestation creation
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn anchor_observe_only_no_key_succeeds_for_empty_room() {
    let obs = fresh_obs(0.0, 0.0);
    let att = create_attestation(b"test", &obs, &AnchorPolicy::observe_only(), None).unwrap();
    assert_eq!(att.algorithm, AnchorAlgorithm::Unsigned);
}

#[test]
fn anchor_attest_mode_no_key_blocks_zero_presence() {
    let obs = fresh_obs(0.0, 0.0);
    let err = create_attestation(b"test", &obs, &AnchorPolicy::attest(), None).unwrap_err();
    assert!(matches!(err, AnchorError::PolicyViolation(_)));
}

#[test]
fn anchor_attest_mode_no_key_produces_valid_unsigned_envelope() {
    let obs = fresh_obs(0.9, 0.8);
    let att = create_attestation(b"handoff:Lumi->Codex:T-019", &obs, &AnchorPolicy::attest(), None).unwrap();
    assert_eq!(att.algorithm, AnchorAlgorithm::Unsigned);
    assert_eq!(att.signature, "unsigned");
    assert_eq!(att.payload_hash.len(), 64);
    assert_eq!(att.observation_hash.len(), 64);
    assert_ne!(att.payload_hash, att.observation_hash);
}

#[test]
fn anchor_attest_with_key_produces_hybrid_algorithm_by_default() {
    let key = AnchorSigningKey::generate();
    let obs = fresh_obs(0.9, 0.8);
    let att = create_attestation(b"signed event", &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();
    assert_eq!(att.algorithm, AnchorAlgorithm::Hybrid);
    assert_ne!(att.signature, "unsigned");
    assert!(att.signature_pq.is_some());
    assert_eq!(att.key_fingerprint, key.fingerprint());
    assert_eq!(att.key_fingerprint_pq.as_deref(), Some(key.fingerprint_pq().as_str()));
}

#[test]
fn anchor_different_payloads_produce_different_payload_hashes() {
    let obs_a = fresh_obs(0.9, 0.8);
    let obs_b = fresh_obs(0.9, 0.8);
    let policy = AnchorPolicy::attest();
    let att_a = create_attestation(b"payload-one", &obs_a, &policy, None).unwrap();
    let att_b = create_attestation(b"payload-two", &obs_b, &policy, None).unwrap();
    assert_ne!(att_a.payload_hash, att_b.payload_hash);
}

#[test]
fn anchor_observation_hash_stable_across_different_payloads() {
    // Same obs → same observation_hash regardless of payload
    let obs = fresh_obs(0.9, 0.8);
    let att_a = create_attestation(b"payload-one", &obs, &AnchorPolicy::observe_only(), None).unwrap();
    let att_b = create_attestation(b"payload-two", &obs, &AnchorPolicy::observe_only(), None).unwrap();
    assert_eq!(att_a.observation_hash, att_b.observation_hash);
}

// ─────────────────────────────────────────────────────────────────────────────
// E. Verification roundtrip
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn anchor_sign_and_verify_roundtrip_succeeds() {
    let key = AnchorSigningKey::generate();
    let vk_bytes = key.verifying_key_bytes();
    let vk_pq_bytes = key.dilithium_public_key_bytes();
    let obs = fresh_obs(0.9, 0.8);
    let payload = b"roundtrip-payload";
    let att = create_attestation(payload, &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();
    assert!(verify_attestation(payload, &obs, &att, &vk_bytes, Some(&vk_pq_bytes)).is_ok());
}

#[test]
fn anchor_verify_fails_on_tampered_payload() {
    let key = AnchorSigningKey::generate();
    let vk_bytes = key.verifying_key_bytes();
    let vk_pq_bytes = key.dilithium_public_key_bytes();
    let obs = fresh_obs(0.9, 0.8);
    let att = create_attestation(b"original", &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();
    let result = verify_attestation(b"tampered", &obs, &att, &vk_bytes, Some(&vk_pq_bytes));
    assert!(result.is_err(), "Tampered payload must fail verification");
}

#[test]
fn anchor_verify_fails_with_wrong_key() {
    let key_a = AnchorSigningKey::generate();
    let key_b = AnchorSigningKey::generate();
    let vk_b_bytes = key_b.verifying_key_bytes();
    let vk_b_pq_bytes = key_b.dilithium_public_key_bytes();
    let obs = fresh_obs(0.9, 0.8);
    let att = create_attestation(b"payload", &obs, &AnchorPolicy::attest(), Some(&key_a)).unwrap();
    let result = verify_attestation(b"payload", &obs, &att, &vk_b_bytes, Some(&vk_b_pq_bytes));
    assert!(result.is_err(), "Verification with wrong key must fail");
}

#[test]
fn anchor_verify_unsigned_returns_error() {
    let key = AnchorSigningKey::generate();
    let vk_bytes = key.verifying_key_bytes();
    let obs = fresh_obs(0.9, 0.8);
    let att = create_attestation(b"payload", &obs, &AnchorPolicy::observe_only(), None).unwrap();
    let result = verify_attestation(b"payload", &obs, &att, &vk_bytes, None);
    assert!(matches!(result, Err(AnchorError::SigningError(_))));
}

// ─────────────────────────────────────────────────────────────────────────────
// F. NDJSON output
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn anchor_ndjson_is_single_line_valid_json() {
    let key = AnchorSigningKey::generate();
    let obs = fresh_obs(0.9, 0.8);
    let att = create_attestation(b"ndjson", &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();
    let line = attestation_to_ndjson(&obs, &att);
    assert!(!line.contains('\n'), "NDJSON must not contain newlines");
    let parsed: serde_json::Value = serde_json::from_str(&line).expect("must be valid JSON");
    assert!(parsed["ts"].is_string());
    assert!(parsed["nonce"].is_string());
    assert!(parsed["signature"].is_string());
    assert!(parsed["key_fingerprint"].is_string());
    assert_ne!(parsed["signature"].as_str().unwrap(), "unsigned");
}

#[test]
fn anchor_ndjson_presence_value_matches_observation() {
    let obs = fresh_obs(0.73, 0.62);
    let att = create_attestation(b"test", &obs, &AnchorPolicy::observe_only(), None).unwrap();
    let line = attestation_to_ndjson(&obs, &att);
    let parsed: serde_json::Value = serde_json::from_str(&line).unwrap();
    let presence = parsed["soma_presence"].as_f64().unwrap();
    assert!((presence - 0.73).abs() < 0.001);
}
