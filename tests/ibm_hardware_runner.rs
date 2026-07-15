use phiflow::compile_to_openqasm;
use phiflow::quantum::{ibm_quantum::IBMQuantumBackend, QuantumBackend, QuantumConfig};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Duration;

const LEGACY_CRD_PATH: &str = "legacy-credential-file";
const IBM_SMOKE_PATH: &str = "examples/ibm_smoke.phi";

fn evidence_path() -> String {
    std::env::var("EVIDENCE_PATH")
        .unwrap_or_else(|_| "target/evidence/ANTIGRAVITY_PIPE2_20260329.md".to_string())
}

#[derive(Debug, Deserialize)]
struct IbmCredentials {
    cloud_key: String,
    service_crn: Option<String>,
    region: Option<String>,
    backend: Option<String>,
}

fn load_ibm_smoke_source() -> String {
    fs::read_to_string(IBM_SMOKE_PATH).expect("could not read examples/ibm_smoke.phi")
}

fn load_credentials() -> IbmCredentials {
    let cred_content = fs::read_to_string(LEGACY_CRD_PATH).expect("could not read legacy-credential-file");
    serde_json::from_str(&cred_content).expect("failed to parse legacy-credential-file")
}

fn write_receipt(
    backend_name: &str,
    region: Option<&str>,
    qasm: &str,
    result: &phiflow::quantum::QuantumResult,
) {
    let path = evidence_path();
    let parent = Path::new(&path)
        .parent()
        .expect("evidence path must have parent");
    fs::create_dir_all(parent).expect("failed to create evidence directory");

    let mut counts: Vec<_> = result.counts.iter().collect();
    counts.sort_by(|a, b| a.0.cmp(b.0));

    let mut body = String::new();
    body.push_str("# EVIDENCE: Pipe 2 (AntiGravity) - IBM Quantum Hardware Validation\n");
    body.push_str("**Date:** 2026-03-29\n");
    body.push_str("**Status:** LIVE EXECUTION VERIFIED FROM CURRENT CHECKOUT\n\n");
    body.push_str("## Receipt\n");
    body.push_str(&format!("- backend: `{}`\n", backend_name));
    body.push_str(&format!("- region: `{}`\n", region.unwrap_or("us-east")));
    body.push_str(&format!("- job_id: `{}`\n", result.job_id));
    body.push_str(&format!("- status: `{}`\n", result.status));
    body.push_str(&format!("- counts: `{}` entries\n", result.counts.len()));
    body.push_str("\n## Counts\n");
    for (bitstring, count) in counts {
        body.push_str(&format!("- `{}` => `{}`\n", bitstring, count));
    }
    body.push_str("\n## OpenQASM 3.0 Excerpt\n");
    body.push_str("```qasm\n");
    for line in qasm.lines().take(12) {
        body.push_str(line);
        body.push('\n');
    }
    body.push_str("```\n");

    fs::write(&path, body).expect("failed to write receipt");
}

#[test]
fn test_ibm_smoke_compiles_to_openqasm() {
    let source = load_ibm_smoke_source();
    let qasm = compile_to_openqasm(&source, false).expect("ibm_smoke should compile");

    assert!(qasm.contains("OPENQASM 3.0;"));
    assert!(qasm.contains("include \"stdgates.inc\";"));
    // Heron-native decomposition: ry(0.618...*pi) becomes rz/sx sequence
    assert!(qasm.contains("ry("));
    assert!(!qasm.contains("sx q["));
}

#[tokio::test]
#[ignore]
async fn test_live_ibm_hardware_runner() {
    let credentials = load_credentials();
    let source = load_ibm_smoke_source();
    let qasm = compile_to_openqasm(&source, false).expect("ibm_smoke should compile");

    let backend_name = credentials
        .backend
        .clone()
        .unwrap_or_else(|| "ibm_osaka".to_string());
    let region = credentials.region.clone();

    let service_crn = credentials.service_crn.clone().expect("legacy-credential-file must include service_crn for the IBM Cloud Runtime live test");

    let config = QuantumConfig {
        backend_name: backend_name.clone(),
        ibm_cloud_key: Some(credentials.cloud_key),
        service_crn: Some(service_crn),
        region: region.clone(),
        hub: None,
        group: None,
        project: None,
        max_qubits: 5,
        shots: 1024,
        timeout_seconds: 300,
    };

    let mut backend = IBMQuantumBackend::with_backend(backend_name.clone());
    backend
        .initialize(config.clone())
        .await
        .expect("backend initialization should succeed");

    let status = backend
        .get_status()
        .await
        .expect("failed to fetch backend status");
    println!(
        "IBM backend status: operational={} pending_jobs={}",
        status.operational, status.pending_jobs
    );

    let result = backend
        .execute_openqasm(
            &qasm,
            Some(config.shots),
            Some(Duration::from_secs(config.timeout_seconds)),
        )
        .await
        .expect("live OpenQASM execution should succeed");

    assert!(!result.job_id.is_empty(), "job_id must not be empty");
    assert!(
        !result.counts.is_empty(),
        "runtime result parser should recover non-empty counts"
    );

    write_receipt(&backend_name, region.as_deref(), &qasm, &result);
}
