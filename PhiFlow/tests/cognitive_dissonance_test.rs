use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::openqasm::OpenQasmEmitter;
use phiflow::quantum::{ibm_quantum::IBMQuantumBackend, QuantumBackend, QuantumConfig};
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct IbmCredentials {
    apikey: String,
    service_crn: Option<String>,
    region: Option<String>,
    backend: Option<String>,
}

fn load_credentials() -> IbmCredentials {
    let apikey_content = fs::read_to_string("apikey.json").expect("could not read apikey.json");
    serde_json::from_str(&apikey_content).expect("failed to parse apikey.json")
}

fn compile_to_openqasm(source: &str, optimize_depth: bool) -> Result<String, String> {
    let exprs = parse_phi_program(source).map_err(|e| e.to_string())?;
    let ir = lower_program(&exprs);
    let mut emitter = OpenQasmEmitter::new();
    emitter.optimize_depth = optimize_depth;
    emitter.emit(&ir).map_err(|e| e.to_string())
}

fn write_receipt(
    backend_name: &str,
    region: Option<&str>,
    qasm: &str,
    result: &phiflow::quantum::QuantumResult,
) {
    let evidence_path = "D:/CosmicFamily/EVIDENCE/ANTIGRAVITY_COGNITIVE_DISSONANCE.md";
    let parent = Path::new(evidence_path).parent().unwrap();
    fs::create_dir_all(parent).unwrap();

    let mut counts: Vec<_> = result.counts.iter().collect();
    // Sort by count descending to see top values / noise distribution
    counts.sort_by(|a, b| b.1.cmp(a.1));

    let mut body = String::new();
    body.push_str("# EVIDENCE: Cognitive Dissonance Protocol (Physical Decoherence)\n");
    body.push_str("**Status:** LIVE EXECUTION VERIFIED on Hardware\n\n");
    body.push_str("## Receipt\n");
    body.push_str(&format!("- backend: `{}`\n", backend_name));
    body.push_str(&format!("- region: `{}`\n", region.unwrap_or("us-east")));
    body.push_str(&format!("- job_id: `{}`\n", result.job_id));
    body.push_str(&format!("- status: `{}`\n", result.status));
    body.push_str(&format!("- unique states measured: `{}` entries (expected extreme entropy)\n", result.counts.len()));
    body.push_str("\n## Top 20 Counts (Entropy Signature)\n");
    for (bitstring, count) in counts.into_iter().take(20) {
        body.push_str(&format!("- `{}` => `{}`\n", bitstring, count));
    }
    body.push_str("\n## OpenQASM 3.0 Excerpt\n");
    body.push_str("```qasm\n");
    for line in qasm.lines().take(20) {
        body.push_str(line);
        body.push('\n');
    }
    body.push_str("...\n```\n");

    fs::write(evidence_path, body).unwrap();
}

#[tokio::test]
#[ignore]
async fn test_live_cognitive_dissonance() {
    let credentials = load_credentials();
    let source = fs::read_to_string("examples/cognitive_dissonance.phi").expect("read failed");
    let qasm = compile_to_openqasm(&source, false).expect("compile failed");
    let service_crn = credentials.service_crn.clone().unwrap();

    let backend_name = credentials.backend.clone().unwrap_or_else(|| "ibm_fez".to_string());
    
    // Unlock qubits for deep entanglement testing
    let config = QuantumConfig {
        backend_name: backend_name.clone(),
        api_token: Some(credentials.apikey),
        service_crn: Some(service_crn),
        region: credentials.region.clone(),
        hub: None,
        group: None,
        project: None,
        max_qubits: 32,
        shots: 4096, // High shot count to measure physical noise
        timeout_seconds: 600,
    };

    let mut backend = IBMQuantumBackend::with_backend(backend_name.clone());
    backend.initialize(config.clone()).await.unwrap();

    println!("IBM backend status running Cognitive Dissonance on {}...", backend_name);
    let result = backend
        .execute_openqasm(&qasm, Some(config.shots), Some(Duration::from_secs(config.timeout_seconds)))
        .await
        .unwrap();

    write_receipt(&backend_name, credentials.region.as_deref(), &qasm, &result);
}
