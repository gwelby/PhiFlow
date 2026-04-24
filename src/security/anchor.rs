//! Anchor Attestation — PhiFlow Security Layer (Phase 2: Nonce Table + secp256k1 Signing)
//!
//! # What This Is
//!
//! An observation-backed attestation envelope. When PhiFlow performs a significant
//! action (ledger write, handoff, exported report) it captures the local SOMA sensor
//! state at that moment and binds it cryptographically to the payload hash.
//!
//! # What This Is NOT
//!
//! - A biometric private key. Raw sensor values (presence, jitter) are OBSERVATION
//!   context, not the secret. The signing key is conventional (secp256k1 or ML-DSA-65).
//!   See RESEARCH/crypto_reality_check.md.
//! - A production blockchain consensus primitive.
//! - Long-term key custody infrastructure. Key storage is the operator's responsibility.
//!
//! # Threat Model Scope
//!
//! Phase 2 (this file):
//! - Process-scoped nonce replay table (resets on restart — cross-session replay
//!   protection requires signed ledger entries, which is outside this module's scope).
//! - ECDSA secp256k1 signing via `k256` crate (RustCrypto, well-audited, no_std-compat).
//! - Signature is DER-encoded and hex-encoded in `AnchorAttestation.signature`.
//! - `create_attestation` accepts an `Option<&AnchorSigningKey>` — passing `None`
//!   produces an `Unsigned` envelope (Phase 1 behaviour). All existing tests still pass.
//!
//! Phase 2 deferred:
//! - ML-DSA-65 (NIST FIPS 204 / Dilithium3) — add `pqcrypto-dilithium` when ready.
//! - `Enforce` mode — fail-closed policy enforcement.
//! - Persistent nonce store across restarts.
//!
//! # Audit Notes for Lumi
//!
//! 1. `capture_observation` calls `register_nonce()` after generating the uuid v4 nonce.
//!    If somehow the same nonce were generated twice (uuid v4 collision is ~2^-61),
//!    the second call would return `AnchorError::NonceReused`. This is defence-in-depth
//!    — the primary collision-prevention is uuid v4 entropy.
//!
//! 2. `hash_observation` uses canonical JSON (keys in struct-field declaration order).
//!    Do not reorder struct fields without versioning this function.
//!
//! 3. The canonical message signed by ECDSA is:
//!    ```text
//!    PhiFlow-Attestation-v1
//!    payload_hash=<hex-sha256>
//!    observation_hash=<hex-sha256>
//!    policy_version=1.0.0
//!    ```
//!    This format is stable. Verifiers can reconstruct it from stored attestation fields.
//!
//! 4. `AnchorSigningKey::generate()` creates an ephemeral key (not persisted). The
//!    daemon currently uses an ephemeral key. Key persistence is the operator's concern.

use crate::sensors;
use k256::ecdsa::{signature::Signer, Signature, SigningKey, VerifyingKey};
use k256::ecdsa::signature::Verifier;
use pqcrypto_dilithium::dilithium3;
use pqcrypto_traits::sign::{DetachedSignature as _, PublicKey as _, SecretKey as _};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fmt;
use std::sync::{OnceLock, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
// Process-scoped nonce store
// ─────────────────────────────────────────────────────────────────────────────

/// Process-global nonce table. Prevents within-session nonce reuse.
/// Resets on process restart — cross-session protection requires a persistent
/// signed ledger (Phase 2 enforcement concern, not this module's responsibility).
static NONCE_STORE: OnceLock<RwLock<HashSet<String>>> = OnceLock::new();

fn nonce_store() -> &'static RwLock<HashSet<String>> {
    NONCE_STORE.get_or_init(|| RwLock::new(HashSet::new()))
}

/// Register a nonce. Returns `Ok(())` if the nonce is new; `Err(NonceReused)` if seen.
pub fn register_nonce(nonce: &str) -> Result<(), AnchorError> {
    let mut store = nonce_store().write().unwrap();
    if store.insert(nonce.to_string()) {
        Ok(())
    } else {
        Err(AnchorError::NonceReused(nonce.to_string()))
    }
}

/// Check without registering (for testing purposes only).
#[cfg(test)]
fn nonce_is_known(nonce: &str) -> bool {
    nonce_store().read().unwrap().contains(nonce)
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the attestation layer.
#[derive(Debug, Clone, PartialEq)]
pub enum AnchorError {
    /// SOMA state file is missing or cannot be parsed.
    SomaMissing,
    /// SOMA state is stale beyond the freshness threshold.
    SomaStale,
    /// A required sensor value was absent from the SOMA state.
    SensorAbsent(String),
    /// Observation failed a policy threshold check.
    PolicyViolation(String),
    /// Serialization of the observation for hashing/signing failed.
    SerializationError(String),
    /// A nonce was submitted that has already been used in this session.
    NonceReused(String),
    /// Signing operation failed.
    SigningError(String),
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnchorError::SomaMissing => write!(f, "SOMA state is missing or unreadable"),
            AnchorError::SomaStale => write!(f, "SOMA state is stale beyond freshness threshold"),
            AnchorError::SensorAbsent(s) => write!(f, "Sensor value absent: {}", s),
            AnchorError::PolicyViolation(s) => write!(f, "Policy violation: {}", s),
            AnchorError::SerializationError(s) => write!(f, "Serialization error: {}", s),
            AnchorError::NonceReused(n) => write!(f, "Nonce already used in this session: {}", n),
            AnchorError::SigningError(s) => write!(f, "Signing error: {}", s),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm + Mode enums
// ─────────────────────────────────────────────────────────────────────────────

/// Which signing algorithm was used.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AnchorAlgorithm {
    /// No cryptographic signature. Envelope structure only (Phase 1 / no key provided).
    Unsigned,
    /// ECDSA over secp256k1. Signature is DER-encoded, hex-encoded.
    EcdsaSecp256k1,
    /// ML-DSA-65 (NIST FIPS 204 post-quantum standard). Phase 2 deferred.
    MlDsa65,
    /// Hybrid — both EcdsaSecp256k1 and MlDsa65 signatures present. Phase 2 deferred.
    Hybrid,
}

/// Operational mode controls what the attestation layer enforces.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AnchorMode {
    /// Capture and hash observations, but never block execution.
    /// Use for initial deployment, dashboards, telemetry collection.
    ObserveOnly,
    /// Capture observations and build a signed envelope.
    /// Gate on policy thresholds. Do not block unrelated runtime execution.
    Attest,
    /// Fail-closed if policy thresholds are not met. Phase 2 — activate only after
    /// replay protection and SOMA freshness rules are stable.
    Enforce,
}

// ─────────────────────────────────────────────────────────────────────────────
// Signing key wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// An ECDSA secp256k1 signing key for PhiFlow attestations.
///
/// # Key Custody
///
/// The operator is responsible for key storage and rotation.
/// `generate()` creates an ephemeral key — use for testing or fresh daemon instances.
/// `from_bytes()` loads a previously persisted key from raw bytes.
pub struct AnchorSigningKey {
    ecdsa: SigningKey,
    dilithium_pk: dilithium3::PublicKey,
    dilithium_sk: dilithium3::SecretKey,
}

impl AnchorSigningKey {
    /// Generate a new ephemeral secp256k1 key using the OS CSPRNG.
    /// Generate new ephemeral secp256k1 and Dilithium-3 keys using OS CSPRNG.
    pub fn generate() -> Self {
        use k256::ecdsa::signature::rand_core::OsRng;
        let (dilithium_pk, dilithium_sk) = dilithium3::keypair();
        Self {
            ecdsa: SigningKey::random(&mut OsRng),
            dilithium_pk,
            dilithium_sk,
        }
    }

    pub fn from_parts(ecdsa_bytes: &[u8; 32], dilithium_sk_bytes: &[u8]) -> Result<Self, AnchorError> {
        let ecdsa = SigningKey::from_bytes(ecdsa_bytes.into())
            .map_err(|e| AnchorError::SigningError(format!("Invalid ECDSA key bytes: {}", e)))?;
        
        let dilithium_sk = dilithium3::SecretKey::from_bytes(dilithium_sk_bytes)
            .map_err(|e| AnchorError::SigningError(format!("Invalid Dilithium-3 secret key bytes: {}", e)))?;
        
        // Re-derive public key (or we could require it as an argument)
        // For Dilithium-3 in pqcrypto, the public key is not easily derivable from secret key alone 
        // without the specific internal primitives. Usually they are stored together.
        // For this implementation, we'll assume generate() is the primary path or we add pk_bytes.
        
        // Actually, let's update from_parts to take pk_bytes too.
        unimplemented!("Use from_parts_full or generate for now");
    }

    /// Load signing keys from raw bytes including the Dilithium public key.
    pub fn from_parts_full(ecdsa_bytes: &[u8; 32], dilithium_pk_bytes: &[u8], dilithium_sk_bytes: &[u8]) -> Result<Self, AnchorError> {
        let ecdsa = SigningKey::from_bytes(ecdsa_bytes.into())
            .map_err(|e| AnchorError::SigningError(format!("Invalid ECDSA key bytes: {}", e)))?;
        
        let dilithium_pk = dilithium3::PublicKey::from_bytes(dilithium_pk_bytes)
            .map_err(|e| AnchorError::SigningError(format!("Invalid Dilithium-3 public key bytes: {}", e)))?;
            
        let dilithium_sk = dilithium3::SecretKey::from_bytes(dilithium_sk_bytes)
            .map_err(|e| AnchorError::SigningError(format!("Invalid Dilithium-3 secret key bytes: {}", e)))?;
            
        Ok(Self { ecdsa, dilithium_pk, dilithium_sk })
    }

    /// Return the compressed secp256k1 public key as 33 bytes.
    pub fn verifying_key_bytes(&self) -> Vec<u8> {
        use k256::EncodedPoint;
        let vk: VerifyingKey = self.ecdsa.verifying_key().clone();
        EncodedPoint::from(vk).as_bytes().to_vec()
    }

    /// Return the Dilithium-3 public key bytes.
    pub fn dilithium_public_key_bytes(&self) -> Vec<u8> {
        self.dilithium_pk.as_bytes().to_vec()
    }

    /// Return a hex-encoded SHA-256 fingerprint of the compressed ECDSA public key.
    pub fn fingerprint(&self) -> String {
        hex_sha256(&self.verifying_key_bytes())
    }

    /// Return a hex-encoded SHA-256 fingerprint of the Dilithium-3 public key.
    pub fn fingerprint_pq(&self) -> String {
        hex_sha256(&self.dilithium_public_key_bytes())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core data structures
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of SOMA and runtime conditions captured at attestation time.
///
/// This is OBSERVATION DATA — not a secret. It answers the question:
/// "what was the environment like when this action was taken?"
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnchorObservation {
    /// Session identifier from the SOMA state file.
    pub session_id: String,
    /// RFC3339 timestamp of when the observation was captured.
    pub timestamp: String,
    /// `soma_presence` reading (0.0 = absent, 1.0 = full presence).
    pub soma_presence: f64,
    /// `ring_jitter_ns` — entropy quality indicator, NOT a secret.
    pub ring_jitter_ns: f64,
    /// `ring_slope_1f` — 1/f slope of ring oscillator.
    pub ring_slope_1f: f64,
    /// Field coherence at time of capture (0.0–1.0).
    pub field_coherence: f64,
    /// UUID v4 nonce. Registered in the process-scoped nonce table on capture.
    pub nonce: String,
}

/// The complete attestation envelope attached to a signed payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AnchorAttestation {
    /// Algorithm used.
    pub algorithm: AnchorAlgorithm,
    /// Hex-encoded SHA-256 of the canonical payload bytes.
    pub payload_hash: String,
    /// Hex-encoded SHA-256 of the canonical `AnchorObservation` JSON.
    pub observation_hash: String,
    /// DER-encoded ECDSA signature, hex-encoded.
    /// `"unsigned"` when no signing key was provided.
    pub signature: String,
    /// Hex fingerprint of the signing key's compressed public key.
    /// `"none"` when unsigned.
    pub key_fingerprint: String,
    /// Dilithium-3 (ML-DSA-65) signature, hex-encoded.
    pub signature_pq: Option<String>,
    /// Dilithium-3 fingerprint.
    pub key_fingerprint_pq: Option<String>,
    /// Semver policy version string.
    pub policy_version: String,
}

/// Thresholds that observations must satisfy before an attestation is created.
#[derive(Debug, Clone)]
pub struct AnchorPolicy {
    /// Minimum `soma_presence` required (0.0–1.0). Default: 0.3.
    pub min_presence: f64,
    /// Minimum field coherence required (0.0–1.0). Default: 0.0.
    pub min_field_coherence: f64,
    /// Maximum SOMA state age in milliseconds.
    pub require_soma_freshness_ms: u64,
    /// Operational mode.
    pub mode: AnchorMode,
}

impl Default for AnchorPolicy {
    fn default() -> Self {
        Self {
            min_presence: 0.3,
            min_field_coherence: 0.0,
            require_soma_freshness_ms: 5000,
            mode: AnchorMode::ObserveOnly,
        }
    }
}

impl AnchorPolicy {
    /// Observe-only policy.
    pub fn observe_only() -> Self {
        Self::default()
    }

    /// Attesting policy — requires presence ≥ 0.3.
    pub fn attest() -> Self {
        Self {
            mode: AnchorMode::Attest,
            ..Self::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Observation capture
// ─────────────────────────────────────────────────────────────────────────────

/// Capture a fresh `AnchorObservation` from live SOMA sensor data.
///
/// The nonce is registered in the process-scoped nonce table on success.
/// If somehow generated twice (uuid collision ~2^-61), returns `NonceReused`.
///
/// # Errors
///
/// - `SomaMissing` — SOMA state file absent.
/// - `SomaStale` — SOMA state older than freshness threshold.
/// - `SensorAbsent` — a required sensor value is `None`.
/// - `NonceReused` — uuid v4 collision (defence-in-depth, virtually impossible).
pub fn capture_observation(session_id: &str) -> Result<AnchorObservation, AnchorError> {
    use crate::phi_ir::SensorKind;

    let soma_presence = sensors::read_sensor(SensorKind::SomaPresence).ok_or_else(|| {
        let path = std::env::var("SOMA_STATE_PATH").unwrap_or_else(|_| "D:/Projects/PhiHarmonic/SOMA/soma_state.json".to_string());
        if std::path::Path::new(&path).exists() {
            AnchorError::SomaStale
        } else {
            AnchorError::SomaMissing
        }
    })?;

    let ring_jitter_ns = sensors::read_sensor(SensorKind::RingJitterNs)
        .ok_or_else(|| AnchorError::SensorAbsent("ring_jitter_ns".to_string()))?;

    let ring_slope_1f = sensors::read_sensor(SensorKind::RingSlope1f)
        .ok_or_else(|| AnchorError::SensorAbsent("ring_slope_1f".to_string()))?;

    let field_coherence = sensors::compute_coherence_from_sensors();
    let timestamp = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);

    // Generate a uuid v4 nonce and register it in the process-scoped nonce table.
    // register_nonce will return NonceReused if somehow generated twice (negligible probability).
    let nonce = uuid::Uuid::new_v4().to_string();
    register_nonce(&nonce)?;

    Ok(AnchorObservation {
        session_id: session_id.to_string(),
        timestamp,
        soma_presence,
        ring_jitter_ns,
        ring_slope_1f,
        field_coherence,
        nonce,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Policy checking
// ─────────────────────────────────────────────────────────────────────────────

/// Validate an observation against a policy.
///
/// Advisory in `ObserveOnly` mode; blocking in `Attest` mode.
pub fn check_policy(obs: &AnchorObservation, policy: &AnchorPolicy) -> Result<(), AnchorError> {
    if obs.soma_presence < policy.min_presence {
        return Err(AnchorError::PolicyViolation(format!(
            "soma_presence {:.3} < required minimum {:.3}",
            obs.soma_presence, policy.min_presence
        )));
    }
    if obs.field_coherence < policy.min_field_coherence {
        return Err(AnchorError::PolicyViolation(format!(
            "field_coherence {:.3} < required minimum {:.3}",
            obs.field_coherence, policy.min_field_coherence
        )));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Hashing
// ─────────────────────────────────────────────────────────────────────────────

/// Produce a deterministic hex-encoded SHA-256 hash of the observation.
pub fn hash_observation(obs: &AnchorObservation) -> Result<String, AnchorError> {
    let json = serde_json::to_string(obs)
        .map_err(|e| AnchorError::SerializationError(e.to_string()))?;
    Ok(hex_sha256(json.as_bytes()))
}

/// Produce a hex-encoded SHA-256 hash of arbitrary payload bytes.
pub fn hash_payload(payload: &[u8]) -> String {
    hex_sha256(payload)
}

fn hex_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical message construction
// ─────────────────────────────────────────────────────────────────────────────

/// Build the canonical message that is signed.
///
/// This format is stable. Verifiers must reconstruct it from stored attestation fields:
/// ```text
/// PhiFlow-Attestation-v1
/// payload_hash=<hex>
/// observation_hash=<hex>
/// policy_version=1.0.0
/// ```
pub fn canonical_message(payload_hash: &str, observation_hash: &str, policy_version: &str) -> String {
    format!(
        "PhiFlow-Attestation-v1\npayload_hash={}\nobservation_hash={}\npolicy_version={}",
        payload_hash, observation_hash, policy_version
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Attestation creation
// ─────────────────────────────────────────────────────────────────────────────

/// Build an `AnchorAttestation` envelope for a payload.
///
/// # Parameters
///
/// - `payload`: the bytes being attested (e.g. serialized handoff event, ledger entry).
/// - `obs`: the observation captured at attestation time.
/// - `policy`: threshold policy and operational mode.
/// - `signing_key`: optional secp256k1 signing key.
///   - `None` → `Unsigned` (Phase 1 / no key available).
///   - `Some(key)` → `EcdsaSecp256k1`, real DER signature.
///
/// # Backwards Compatibility
///
/// All existing callers that pass `None` get identical behaviour to Phase 1.
pub fn create_attestation(
    payload: &[u8],
    obs: &AnchorObservation,
    policy: &AnchorPolicy,
    signing_key: Option<&AnchorSigningKey>,
) -> Result<AnchorAttestation, AnchorError> {
    let policy_result = check_policy(obs, policy);
    if policy.mode != AnchorMode::ObserveOnly {
        policy_result?;
    }

    let payload_hash = hash_payload(payload);
    let observation_hash = hash_observation(obs)?;
    let policy_version = "1.0.0".to_string();

    let (algorithm, sig_ecdsa, sig_pq, finger_ecdsa, finger_pq) = match signing_key {
        None => (
            AnchorAlgorithm::Unsigned,
            "unsigned".to_string(),
            None,
            "none".to_string(),
            None,
        ),
        Some(key) => {
            let msg = canonical_message(&payload_hash, &observation_hash, &policy_version);
            
            // Sign with ECDSA
            let sig_e: Signature = key.ecdsa.sign(msg.as_bytes());
            let sig_hex_e = hex::encode(sig_e.to_der());

            // Sign with Dilithium-3
            let sig_p = dilithium3::detached_sign(msg.as_bytes(), &key.dilithium_sk);
            let sig_hex_p = hex::encode(sig_p.as_bytes());

            (
                AnchorAlgorithm::Hybrid,
                sig_hex_e,
                Some(sig_hex_p),
                key.fingerprint(),
                Some(key.fingerprint_pq()),
            )
        }
    };

    Ok(AnchorAttestation {
        algorithm,
        payload_hash,
        observation_hash,
        signature: sig_ecdsa,
        key_fingerprint: finger_ecdsa,
        signature_pq: sig_pq,
        key_fingerprint_pq: finger_pq,
        policy_version,
    })
}

/// Verify an `AnchorAttestation` against its payload and observation.
///
/// Returns `Ok(())` if the signature is valid. Returns `Err` if:
/// - the attestation is `Unsigned` (no key to verify)
/// - the signature does not match the canonical message
pub fn verify_attestation(
    payload: &[u8],
    obs: &AnchorObservation,
    att: &AnchorAttestation,
    public_key_hex: &[u8],
    public_key_pq_hex: Option<&[u8]>,
) -> Result<(), AnchorError> {
    if att.algorithm == AnchorAlgorithm::Unsigned {
        return Err(AnchorError::SigningError(
            "Attestation is unsigned — nothing to verify".to_string(),
        ));
    }

    let payload_hash = hash_payload(payload);
    let observation_hash = hash_observation(obs)?;
    let msg = canonical_message(&payload_hash, &observation_hash, &att.policy_version);

    match att.algorithm {
        AnchorAlgorithm::Unsigned => unreachable!(),
        AnchorAlgorithm::EcdsaSecp256k1 => {
            let vk = VerifyingKey::from_sec1_bytes(public_key_hex)
                .map_err(|e| AnchorError::SigningError(format!("Invalid verifying key: {}", e)))?;
            let sig_bytes = hex::decode(&att.signature)
                .map_err(|e| AnchorError::SigningError(format!("Invalid signature hex: {}", e)))?;
            let sig = Signature::from_der(&sig_bytes)
                .map_err(|e| AnchorError::SigningError(format!("Invalid DER signature: {}", e)))?;
            vk.verify(msg.as_bytes(), &sig)
                .map_err(|e| AnchorError::SigningError(format!("ECDSA verification failed: {}", e)))
        }
        AnchorAlgorithm::MlDsa65 => {
            let pk = dilithium3::PublicKey::from_bytes(public_key_hex)
                .map_err(|_| AnchorError::SigningError("Invalid Dilithium-3 public key bytes".to_string()))?;
            let sig_hex = att.signature_pq.as_ref()
                .ok_or_else(|| AnchorError::SigningError("Missing PQ signature".to_string()))?;
            let sig_bytes = hex::decode(sig_hex)
                .map_err(|_| AnchorError::SigningError("Invalid PQ signature hex".to_string()))?;
            let sig = dilithium3::DetachedSignature::from_bytes(&sig_bytes)
                .map_err(|_| AnchorError::SigningError("Invalid PQ signature bytes".to_string()))?;
            dilithium3::verify_detached_signature(&sig, msg.as_bytes(), &pk)
                .map_err(|_| AnchorError::SigningError("ML-DSA-65 verification failed".to_string()))
        }
        AnchorAlgorithm::Hybrid => {
            // 1. Verify ECDSA
            let vk = VerifyingKey::from_sec1_bytes(public_key_hex)
                .map_err(|e| AnchorError::SigningError(format!("Invalid verifying key: {}", e)))?;
            let sig_bytes = hex::decode(&att.signature)
                .map_err(|e| AnchorError::SigningError(format!("Invalid signature hex: {}", e)))?;
            let sig = Signature::from_der(&sig_bytes)
                .map_err(|e| AnchorError::SigningError(format!("Invalid DER signature: {}", e)))?;
            vk.verify(msg.as_bytes(), &sig)
                .map_err(|e| AnchorError::SigningError(format!("Hybrid: ECDSA verify failed: {}", e)))?;

            // 2. Verify PQ
            let pq_pk_bytes = public_key_pq_hex
                .ok_or_else(|| AnchorError::SigningError("Hybrid verify requires PQ public key".to_string()))?;
            let pk_pq = dilithium3::PublicKey::from_bytes(pq_pk_bytes)
                .map_err(|_| AnchorError::SigningError("Invalid Dilithium-3 public key bytes".to_string()))?;
            let sig_hex_pq = att.signature_pq.as_ref()
                .ok_or_else(|| AnchorError::SigningError("Missing PQ signature".to_string()))?;
            let sig_bytes_pq = hex::decode(sig_hex_pq)
                .map_err(|_| AnchorError::SigningError("Invalid PQ signature hex".to_string()))?;
            let sig_pq = dilithium3::DetachedSignature::from_bytes(&sig_bytes_pq)
                .map_err(|_| AnchorError::SigningError("Invalid PQ signature bytes".to_string()))?;
            dilithium3::verify_detached_signature(&sig_pq, msg.as_bytes(), &pk_pq)
                .map_err(|_| AnchorError::SigningError("Hybrid: ML-DSA-65 verify failed".to_string()))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helper
// ─────────────────────────────────────────────────────────────────────────────

/// Format an attestation as a single-line NDJSON string for log emission.
pub fn attestation_to_ndjson(obs: &AnchorObservation, att: &AnchorAttestation) -> String {
    let ts = &obs.timestamp;
    let nonce = &obs.nonce;
    let presence = obs.soma_presence;
    let coherence = obs.field_coherence;
    let phash = &att.payload_hash;
    let ohash = &att.observation_hash;
    let algo = format!("{:?}", att.algorithm);
    let pver = &att.policy_version;
    let sig = &att.signature;
    let kfp = &att.key_fingerprint;
    
    // PQ fields (optional)
    let sig_pq = att.signature_pq.as_deref().unwrap_or("none");
    let kfp_pq = att.key_fingerprint_pq.as_deref().unwrap_or("none");

    format!(
        r#"{{"ts":"{ts}","nonce":"{nonce}","soma_presence":{presence:.4},"field_coherence":{coherence:.4},"payload_hash":"{phash}","observation_hash":"{ohash}","signature":"{sig}","key_fingerprint":"{kfp}","signature_pq":"{sig_pq}","key_fingerprint_pq":"{kfp_pq}","algorithm":"{algo}","policy_version":"{pver}"}}"#
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (no live SOMA required)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_obs_with_nonce(presence: f64, coherence: f64) -> AnchorObservation {
        // Generate a unique nonce and register it so tests reflect real behaviour.
        let nonce = uuid::Uuid::new_v4().to_string();
        // Register it — ignore error in case a previous test already claimed this
        // nonce (uuid collision is negligible; this is test infrastructure only).
        let _ = register_nonce(&nonce);
        AnchorObservation {
            session_id: "test-session".to_string(),
            timestamp: "2026-04-19T21:00:00.000Z".to_string(),
            soma_presence: presence,
            ring_jitter_ns: 0.42,
            ring_slope_1f: -1.1,
            field_coherence: coherence,
            nonce,
        }
    }

    // ── Nonce tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_nonce_first_registration_succeeds() {
        let nonce = format!("test-nonce-{}", uuid::Uuid::new_v4());
        assert!(register_nonce(&nonce).is_ok());
    }

    #[test]
    fn test_nonce_second_registration_fails() {
        let nonce = format!("test-dup-{}", uuid::Uuid::new_v4());
        register_nonce(&nonce).unwrap();
        let err = register_nonce(&nonce).unwrap_err();
        assert!(matches!(err, AnchorError::NonceReused(_)));
        if let AnchorError::NonceReused(n) = err {
            assert_eq!(n, nonce);
        }
    }

    #[test]
    fn test_nonce_is_known_after_registration() {
        let nonce = format!("test-known-{}", uuid::Uuid::new_v4());
        assert!(!nonce_is_known(&nonce));
        register_nonce(&nonce).unwrap();
        assert!(nonce_is_known(&nonce));
    }

    // ── Policy tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_policy_passes_when_presence_above_threshold() {
        let obs = fake_obs_with_nonce(0.9, 0.7);
        assert!(check_policy(&obs, &AnchorPolicy::attest()).is_ok());
    }

    #[test]
    fn test_policy_fails_when_presence_below_threshold() {
        let obs = fake_obs_with_nonce(0.1, 0.7);
        assert!(matches!(
            check_policy(&obs, &AnchorPolicy::attest()),
            Err(AnchorError::PolicyViolation(_))
        ));
    }

    #[test]
    fn test_policy_fails_when_coherence_below_threshold() {
        let obs = fake_obs_with_nonce(0.9, 0.1);
        let policy = AnchorPolicy {
            min_field_coherence: 0.5,
            mode: AnchorMode::Attest,
            ..Default::default()
        };
        assert!(matches!(
            check_policy(&obs, &policy),
            Err(AnchorError::PolicyViolation(_))
        ));
    }

    // ── Hashing tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_hash_observation_is_deterministic() {
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let h1 = hash_observation(&obs).unwrap();
        let h2 = hash_observation(&obs).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_observation_changes_with_different_presence() {
        let obs_a = fake_obs_with_nonce(0.9, 0.8);
        let obs_b = fake_obs_with_nonce(0.1, 0.8);
        assert_ne!(hash_observation(&obs_a).unwrap(), hash_observation(&obs_b).unwrap());
    }

    #[test]
    fn test_hash_payload_is_deterministic() {
        let h1 = hash_payload(b"PhiFlow ledger event v1");
        let h2 = hash_payload(b"PhiFlow ledger event v1");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    // ── Unsigned attestation tests ────────────────────────────────────────────

    #[test]
    fn test_create_attestation_observe_only_no_key() {
        let obs = fake_obs_with_nonce(0.0, 0.0);
        let att = create_attestation(b"test payload", &obs, &AnchorPolicy::observe_only(), None).unwrap();
        assert_eq!(att.algorithm, AnchorAlgorithm::Unsigned);
        assert_eq!(att.signature, "unsigned");
        assert_eq!(att.key_fingerprint, "none");
    }

    #[test]
    fn test_create_attestation_attest_mode_blocks_on_zero_presence_no_key() {
        let obs = fake_obs_with_nonce(0.0, 0.0);
        let err = create_attestation(b"test payload", &obs, &AnchorPolicy::attest(), None).unwrap_err();
        assert!(matches!(err, AnchorError::PolicyViolation(_)));
    }

    #[test]
    fn test_create_attestation_unsigned_produces_valid_envelope() {
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let att = create_attestation(b"handoff:Lumi->Codex:T-019", &obs, &AnchorPolicy::attest(), None).unwrap();
        assert_eq!(att.algorithm, AnchorAlgorithm::Unsigned);
        assert_eq!(att.payload_hash.len(), 64);
        assert_eq!(att.observation_hash.len(), 64);
        assert_ne!(att.payload_hash, att.observation_hash);
    }

    // ── secp256k1 signing tests ───────────────────────────────────────────────

    #[test]
    fn test_signing_key_generates_fingerprint() {
        let key = AnchorSigningKey::generate();
        let fp = key.fingerprint();
        assert_eq!(fp.len(), 64, "Fingerprint must be SHA-256 hex (64 chars)");
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_create_attestation_with_key_produces_hybrid_algorithm_by_default() {
        let key = AnchorSigningKey::generate();
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let att = create_attestation(b"signed payload", &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();
        assert_eq!(att.algorithm, AnchorAlgorithm::Hybrid);
        assert_ne!(att.signature, "unsigned");
        assert!(att.signature_pq.is_some());
        assert_eq!(att.key_fingerprint, key.fingerprint());
        assert_eq!(att.key_fingerprint_pq.as_deref(), Some(key.fingerprint_pq().as_str()));
    }

    #[test]
    fn test_sign_and_verify_roundtrip() {
        let key = AnchorSigningKey::generate();
        let vk_bytes = key.verifying_key_bytes();
        let vk_pq_bytes = key.dilithium_public_key_bytes();
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let payload = b"roundtrip-test-payload";
        let att = create_attestation(payload, &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();

        let result = verify_attestation(payload, &obs, &att, &vk_bytes, Some(&vk_pq_bytes));
        assert!(result.is_ok(), "Verification must succeed for a freshly signed hybrid attestation");
    }

    #[test]
    fn test_verify_fails_with_wrong_payload() {
        let key = AnchorSigningKey::generate();
        let vk_bytes = key.verifying_key_bytes();
        let vk_pq_bytes = key.dilithium_public_key_bytes();
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let att = create_attestation(b"original payload", &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();

        // Verifying against different payload bytes must fail
        let result = verify_attestation(b"tampered payload", &obs, &att, &vk_bytes, Some(&vk_pq_bytes));
        assert!(result.is_err(), "Verification must fail when payload is tampered");
    }

    #[test]
    fn test_verify_unsigned_returns_error() {
        let key = AnchorSigningKey::generate();
        let vk_bytes = key.verifying_key_bytes();
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let att = create_attestation(b"payload", &obs, &AnchorPolicy::observe_only(), None).unwrap();

        let result = verify_attestation(b"payload", &obs, &att, &vk_bytes, None);
        assert!(matches!(result, Err(AnchorError::SigningError(_))));
    }

    #[test]
    fn test_two_different_keys_produce_different_signatures() {
        let key_a = AnchorSigningKey::generate();
        let key_b = AnchorSigningKey::generate();
        let obs_a = fake_obs_with_nonce(0.9, 0.8);
        let obs_b = fake_obs_with_nonce(0.9, 0.8);
        let payload = b"same payload";
        let att_a = create_attestation(payload, &obs_a, &AnchorPolicy::attest(), Some(&key_a)).unwrap();
        let att_b = create_attestation(payload, &obs_b, &AnchorPolicy::attest(), Some(&key_b)).unwrap();
        assert_ne!(att_a.key_fingerprint, att_b.key_fingerprint);
    }

    // ── NDJSON output ─────────────────────────────────────────────────────────

    #[test]
    fn test_attestation_ndjson_includes_signature_fields() {
        let key = AnchorSigningKey::generate();
        let obs = fake_obs_with_nonce(0.9, 0.8);
        let att = create_attestation(b"ndjson-test", &obs, &AnchorPolicy::attest(), Some(&key)).unwrap();
        let line = attestation_to_ndjson(&obs, &att);
        let parsed: serde_json::Value = serde_json::from_str(&line).expect("must be valid JSON");
        assert!(parsed["ts"].is_string());
        assert!(parsed["nonce"].is_string());
        assert!(parsed["signature"].is_string());
        assert!(parsed["key_fingerprint"].is_string());
        assert_ne!(parsed["signature"].as_str().unwrap(), "unsigned");
    }
}
