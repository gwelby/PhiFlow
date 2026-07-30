//! pqc_tool — Post-Quantum Cryptography (PQC) / Quantum-Safe Hybrid Signature CLI Tool
//!
//! Provides CLI access to the PhiFlow `security::anchor` module for ECDSA (secp256k1) + ML-DSA-65 (Dilithium3)
//! key generation, payload signing, and attestation verification.

use phiflow::security::anchor::{
    create_attestation, verify_attestation, AnchorAttestation, AnchorObservation,
    AnchorPolicy, AnchorSigningKey,
};
use std::env;
use std::error::Error;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let cmd = &args[1];
    match cmd.as_str() {
        "keygen" => {
            if let Err(e) = handle_keygen() {
                eprintln!(r#"{{"error": "Keygen failed: {}"}}"#, e);
                std::process::exit(1);
            }
        }
        "sign" => {
            if let Err(e) = handle_sign(&args[2..]) {
                eprintln!(r#"{{"error": "Signing failed: {}"}}"#, e);
                std::process::exit(1);
            }
        }
        "verify" => {
            if let Err(e) = handle_verify(&args[2..]) {
                println!(r#"{{"valid": false, "error": "Verification failed: {}"}}"#, e);
                std::process::exit(1);
            }
        }
        _ => {
            eprintln!("Unknown command: {}", cmd);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Usage: pqc_tool <command> [args]");
    eprintln!("Commands:");
    eprintln!("  keygen");
    eprintln!("  sign --ecdsa-sk <hex> --dilithium-sk <hex> --dilithium-pk <hex> --payload <hex> --obs <json>");
    eprintln!("  verify --ecdsa-vk <hex> --dilithium-pk <hex> --payload <hex> --obs <json> --attestation <json>");
}

fn handle_keygen() -> Result<(), Box<dyn Error>> {
    let _key = AnchorSigningKey::generate();
    
    // In k256, the internal SigningKey's to_bytes() gets us the 32 secret bytes.
    // However, to avoid compiler version mismatches on to_bytes(), let's re-serialize or reflect from our own parts.
    // Wait, let's look at the structure of `AnchorSigningKey` we inspected:
    // It has `verifying_key_bytes()`, `dilithium_public_key_bytes()`, `fingerprint()`, `fingerprint_pq()`.
    // Wait, we need the raw ecdsa private bytes and dilithium secret bytes.
    // Since AnchorSigningKey doesn't expose the private key fields as public, let's check how we can get them.
    // Wait! Let's check `k256` features and if we can generate them ourselves in `pqc_tool` and then construct `AnchorSigningKey`
    // using `from_parts_full`.
    // That is brilliant! By generating them ourselves, we have 100% control over exporting them as hex, and then we verify 
    // compatibility by loading them back!
    use k256::ecdsa::signature::rand_core::OsRng;
    use k256::ecdsa::SigningKey;
    use pqcrypto_dilithium::dilithium3;
    use pqcrypto_traits::sign::{SecretKey as _, PublicKey as _};

    // 1. Generate ECDSA key
    let ecdsa_sk = SigningKey::random(&mut OsRng);
    let ecdsa_sk_bytes = ecdsa_sk.to_bytes();
    let ecdsa_vk_bytes = ecdsa_sk.verifying_key().to_sec1_bytes();

    // 2. Generate Dilithium3 key
    let (dil_pk, dil_sk) = dilithium3::keypair();

    // Verify compatibility by loading them into AnchorSigningKey
    let ecdsa_array: &[u8; 32] = ecdsa_sk_bytes.as_slice().try_into()
        .map_err(|_| "Failed to cast ECDSA private key to 32 bytes")?;
    let _signing_key = AnchorSigningKey::from_parts_full(
        ecdsa_array,
        dil_pk.as_bytes(),
        dil_sk.as_bytes()
    ).map_err(|e| format!("{:?}", e))?;

    // Output JSON
    println!(
        r#"{{"ecdsa_sk":"{}","ecdsa_vk":"{}","dilithium_sk":"{}","dilithium_pk":"{}","key_fingerprint":"{}","key_fingerprint_pq":"{}"}}"#,
        hex::encode(ecdsa_sk_bytes),
        hex::encode(ecdsa_vk_bytes),
        hex::encode(dil_sk.as_bytes()),
        hex::encode(dil_pk.as_bytes()),
        _signing_key.fingerprint(),
        _signing_key.fingerprint_pq()
    );

    Ok(())
}

fn handle_sign(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut ecdsa_sk_hex = None;
    let mut dil_sk_hex = None;
    let mut dil_pk_hex = None;
    let mut payload_hex = None;
    let mut obs_json = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ecdsa-sk" => {
                ecdsa_sk_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--dilithium-sk" => {
                dil_sk_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--dilithium-pk" => {
                dil_pk_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--payload" => {
                payload_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--obs" => {
                obs_json = Some(&args[i + 1]);
                i += 2;
            }
            _ => {
                return Err(format!("Unknown sign argument: {}", args[i]).into());
            }
        }
    }

    let ecdsa_sk_bytes = hex::decode(ecdsa_sk_hex.ok_or("Missing --ecdsa-sk")?)?;
    let dil_sk_bytes = hex::decode(dil_sk_hex.ok_or("Missing --dilithium-sk")?)?;
    let dil_pk_bytes = hex::decode(dil_pk_hex.ok_or("Missing --dilithium-pk")?)?;
    let payload_bytes = hex::decode(payload_hex.ok_or("Missing --payload")?)?;
    let obs_str = obs_json.ok_or("Missing --obs")?;

    let ecdsa_array: &[u8; 32] = ecdsa_sk_bytes.as_slice().try_into()
        .map_err(|_| "ECDSA secret key must be exactly 32 bytes")?;

    // Load key using from_parts_full
    let signing_key = AnchorSigningKey::from_parts_full(
        ecdsa_array,
        &dil_pk_bytes,
        &dil_sk_bytes,
    ).map_err(|e| format!("{:?}", e))?;

    // Parse observation
    let obs: AnchorObservation = serde_json::from_str(obs_str)?;

    // Create attestation using AnchorPolicy with ObserveOnly so we don't do live SOMA checks in CLI
    let att = create_attestation(
        &payload_bytes,
        &obs,
        &AnchorPolicy::observe_only(),
        Some(&signing_key),
    ).map_err(|e| format!("{:?}", e))?;

    // Output JSON
    println!("{}", serde_json::to_string(&att)?);
    Ok(())
}

fn handle_verify(args: &[String]) -> Result<(), Box<dyn Error>> {
    let mut ecdsa_vk_hex = None;
    let mut dil_pk_hex = None;
    let mut payload_hex = None;
    let mut obs_json = None;
    let mut att_json = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ecdsa-vk" => {
                ecdsa_vk_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--dilithium-pk" => {
                dil_pk_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--payload" => {
                payload_hex = Some(&args[i + 1]);
                i += 2;
            }
            "--obs" => {
                obs_json = Some(&args[i + 1]);
                i += 2;
            }
            "--attestation" => {
                att_json = Some(&args[i + 1]);
                i += 2;
            }
            _ => {
                return Err(format!("Unknown verify argument: {}", args[i]).into());
            }
        }
    }

    let ecdsa_vk_bytes = hex::decode(ecdsa_vk_hex.ok_or("Missing --ecdsa-vk")?)?;
    let dil_pk_bytes = hex::decode(dil_pk_hex.ok_or("Missing --dilithium-pk")?)?;
    let payload_bytes = hex::decode(payload_hex.ok_or("Missing --payload")?)?;
    let obs_str = obs_json.ok_or("Missing --obs")?;
    let att_str = att_json.ok_or("Missing --attestation")?;

    // Parse observation and attestation
    let obs: AnchorObservation = serde_json::from_str(obs_str)?;
    let att: AnchorAttestation = serde_json::from_str(att_str)?;

    // Verify attestation
    verify_attestation(
        &payload_bytes,
        &obs,
        &att,
        &ecdsa_vk_bytes,
        Some(&dil_pk_bytes),
    ).map_err(|e| format!("{:?}", e))?;

    println!(r#"{{"valid": true}}"#);
    Ok(())
}
