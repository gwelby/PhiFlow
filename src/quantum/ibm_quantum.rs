// PhiFlow IBM Quantum Backend - Real IBM Quantum computer integration
// Provides connectivity to IBM Quantum Experience via REST API

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{debug, info, warn};

use super::types::*;

const LEGACY_BASE_URL: &str = "https://api.quantum-computing.ibm.com";
const RUNTIME_BASE_URL_US: &str = "https://quantum.cloud.ibm.com/api/v1";
const RUNTIME_BASE_URL_EU: &str = "https://eu-de.quantum.cloud.ibm.com/api/v1";
const IBM_API_VERSION: &str = "2026-02-15";

pub struct IBMQuantumBackend {
    client: Client,
    api_token: Option<String>,
    service_crn: Option<String>,
    region: Option<String>,
    base_url: String,
    hub: Option<String>,
    group: Option<String>,
    project: Option<String>,
    backend_name: String,
    capabilities: Option<QuantumCapabilities>,
    shots: u32,
    timeout_seconds: u64,
}

impl IBMQuantumBackend {
    pub fn new() -> Self {
        IBMQuantumBackend {
            client: Client::new(),
            api_token: None,
            service_crn: None,
            region: None,
            base_url: LEGACY_BASE_URL.to_string(),
            hub: None,
            group: None,
            project: None,
            backend_name: "ibmq_qasm_simulator".to_string(),
            capabilities: None,
            shots: 1024,
            timeout_seconds: 300,
        }
    }

    pub fn with_backend(backend_name: String) -> Self {
        let mut backend = Self::new();
        backend.backend_name = backend_name;
        backend
    }

    fn runtime_base_url(region: Option<&str>) -> String {
        match region.unwrap_or("us-east") {
            "eu-de" => RUNTIME_BASE_URL_EU.to_string(),
            _ => RUNTIME_BASE_URL_US.to_string(),
        }
    }

    fn is_runtime_api(&self) -> bool {
        self.service_crn.is_some()
    }

    fn scrubbed_excerpt(body: &str) -> String {
        let trimmed = body.replace('\n', " ").replace('\r', " ");
        let excerpt: String = trimmed.chars().take(300).collect();
        excerpt
    }

    async fn response_error(
        context: &str,
        response: reqwest::Response,
        auth: bool,
    ) -> QuantumError {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let message = format!("{}: {} {}", context, status, Self::scrubbed_excerpt(&body));
        if auth {
            QuantumError::AuthError { message }
        } else {
            QuantumError::BackendError { message }
        }
    }

    async fn authenticate(&self) -> QuantumResult2<String> {
        let token = self
            .api_token
            .as_ref()
            .ok_or_else(|| QuantumError::AuthError {
                message: "No API token provided".to_string(),
            })?;

        if self.is_runtime_api() {
            let response = self
                .client
                .post("https://iam.cloud.ibm.com/identity/token")
                .header("Content-Type", "application/x-www-form-urlencoded")
                .header("Accept", "application/json")
                .form(&[
                    ("grant_type", "urn:ietf:params:oauth:grant-type:apikey"),
                    ("apikey", token.as_str()),
                ])
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(
                    Self::response_error("IBM IAM authentication failed", response, true).await,
                );
            }

            let data: Value = response.json().await?;
            let access_token =
                data["access_token"]
                    .as_str()
                    .ok_or_else(|| QuantumError::AuthError {
                        message: "IBM IAM response did not contain access_token".to_string(),
                    })?;

            info!(
                "IBM Cloud IAM auth successful for region {}",
                self.region.as_deref().unwrap_or("us-east")
            );
            Ok(access_token.to_string())
        } else {
            let response = self
                .client
                .post(format!("{}/api/Network/login", self.base_url))
                .json(&json!({ "apiToken": token }))
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(Self::response_error(
                    "Legacy IBM Quantum authentication failed",
                    response,
                    true,
                )
                .await);
            }

            let data: Value = response.json().await?;
            let access_token = data["id"].as_str().ok_or_else(|| QuantumError::AuthError {
                message: "Legacy IBM auth response did not contain token id".to_string(),
            })?;
            Ok(access_token.to_string())
        }
    }

    fn build_request(
        &self,
        access_token: &str,
        method: reqwest::Method,
        path: &str,
    ) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut req = self.client.request(method, &url);

        if let Some(crn) = &self.service_crn {
            req = req
                .header("Authorization", format!("Bearer {}", access_token))
                .header("Service-CRN", crn)
                .header("IBM-API-Version", IBM_API_VERSION);
        } else {
            req = req.header("X-Access-Token", access_token);
        }

        req
    }

    async fn get_runtime_backend_resource(
        &self,
        access_token: &str,
        resource: &str,
    ) -> QuantumResult2<Value> {
        let path = format!("/backends/{}/{}", self.backend_name, resource);
        let response = self
            .build_request(access_token, reqwest::Method::GET, &path)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            Err(Self::response_error(
                "Failed to get IBM runtime backend resource",
                response,
                false,
            )
            .await)
        }
    }

    async fn get_backend_info(&self, access_token: &str) -> QuantumResult2<Value> {
        if self.is_runtime_api() {
            let configuration = self
                .get_runtime_backend_resource(access_token, "configuration")
                .await?;
            let status = self
                .get_runtime_backend_resource(access_token, "status")
                .await?;
            Ok(json!({
                "configuration": configuration,
                "status": status
            }))
        } else {
            let response = self
                .build_request(
                    access_token,
                    reqwest::Method::GET,
                    &format!("/api/Backends/{}", self.backend_name),
                )
                .send()
                .await?;

            if response.status().is_success() {
                Ok(response.json().await?)
            } else {
                Err(Self::response_error("Failed to get IBM backend info", response, false).await)
            }
        }
    }

    fn capability_payload<'a>(&self, backend_info: &'a Value) -> &'a Value {
        if self.is_runtime_api() {
            backend_info.get("configuration").unwrap_or(backend_info)
        } else {
            backend_info
        }
    }

    fn status_payload<'a>(&self, backend_info: &'a Value) -> &'a Value {
        if self.is_runtime_api() {
            backend_info.get("status").unwrap_or(backend_info)
        } else {
            backend_info
        }
    }

    fn extract_max_qubits(&self, backend_info: &Value) -> u32 {
        let config = self.capability_payload(backend_info);
        config["nQubits"]
            .as_u64()
            .or_else(|| config["n_qubits"].as_u64())
            .or_else(|| config["num_qubits"].as_u64())
            .or_else(|| config["numQubits"].as_u64())
            .unwrap_or(5) as u32
    }

    fn extract_basis_gates(&self, backend_info: &Value) -> Vec<String> {
        let config = self.capability_payload(backend_info);
        for key in [
            "basisGates",
            "basis_gates",
            "gates",
            "supported_instructions",
        ] {
            if let Some(values) = config[key].as_array() {
                let gates: Vec<String> = values
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                if !gates.is_empty() {
                    return gates;
                }
            }
        }

        vec!["h".to_string(), "x".to_string(), "cx".to_string()]
    }

    fn extract_backend_status(&self, backend_info: &Value) -> BackendStatus {
        let payload = self.status_payload(backend_info);
        let raw_status = payload["status"]
            .as_str()
            .or_else(|| payload["state"].as_str())
            .or_else(|| {
                payload["operational"]
                    .as_bool()
                    .map(|v| if v { "ONLINE" } else { "OFFLINE" })
            })
            .unwrap_or("Unknown");
        let operational = matches!(
            raw_status.to_uppercase().as_str(),
            "ONLINE" | "ACTIVE" | "AVAILABLE" | "TRUE"
        );
        let pending_jobs = payload["lengthQueue"]
            .as_u64()
            .or_else(|| payload["length_queue"].as_u64())
            .or_else(|| payload["pending_jobs"].as_u64())
            .unwrap_or(0) as u32;

        BackendStatus {
            operational,
            pending_jobs,
            queue_length: pending_jobs,
            status_msg: payload["statusMsg"]
                .as_str()
                .or_else(|| payload["status_msg"].as_str())
                .or_else(|| payload["message"].as_str())
                .or_else(|| payload["status"].as_str())
                .or_else(|| payload["state"].as_str())
                .unwrap_or("Unknown")
                .to_string(),
            last_update: chrono::Utc::now().to_rfc3339(),
        }
    }

    fn runtime_job_payload(&self, openqasm: &str, shots: u32) -> Value {
        json!({
            "program_id": "sampler",
            "backend": self.backend_name,
            "params": {
                "pubs": [[openqasm, [], shots]],
                "version": 2,
                "support_qiskit": false
            }
        })
    }

    async fn submit_openqasm_job(
        &self,
        access_token: &str,
        openqasm: &str,
        shots: u32,
    ) -> QuantumResult2<String> {
        let response = if self.is_runtime_api() {
            self.build_request(access_token, reqwest::Method::POST, "/jobs")
                .header("Content-Type", "application/json")
                .json(&self.runtime_job_payload(openqasm, shots))
                .send()
                .await?
        } else {
            self.build_request(access_token, reqwest::Method::POST, "/api/Jobs")
                .header("Content-Type", "application/json")
                .json(&json!({
                    "qObject": {
                        "qobj_id": uuid::Uuid::new_v4().to_string(),
                        "type": "QASM",
                        "schema_version": "1.3.0",
                        "experiments": [{
                            "name": "phiflow_experiment",
                            "qasm": openqasm
                        }],
                        "backend_name": self.backend_name,
                        "shots": shots
                    }
                }))
                .send()
                .await?
        };

        if !response.status().is_success() {
            return Err(Self::response_error("IBM job submission failed", response, false).await);
        }

        let result: Value = response.json().await?;
        let job_id = result["id"]
            .as_str()
            .or_else(|| result["job_id"].as_str())
            .ok_or_else(|| QuantumError::BackendError {
                message: "IBM job submission response did not contain a job ID".to_string(),
            })?;

        info!("IBM job submitted: {}", job_id);
        Ok(job_id.to_string())
    }

    async fn fetch_job_results(&self, access_token: &str, job_id: &str) -> QuantumResult2<Value> {
        let path = if self.is_runtime_api() {
            format!("/jobs/{}/results", job_id)
        } else {
            format!("/api/Jobs/{}", job_id)
        };

        let response = self
            .build_request(access_token, reqwest::Method::GET, &path)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            Err(Self::response_error("Failed to fetch IBM job results", response, false).await)
        }
    }

    async fn wait_for_job(
        &self,
        access_token: &str,
        job_id: &str,
        timeout: Duration,
    ) -> QuantumResult2<Value> {
        let start_time = Instant::now();
        let check_interval = Duration::from_secs(3);
        let path = if self.is_runtime_api() {
            format!("/jobs/{}", job_id)
        } else {
            format!("/api/Jobs/{}", job_id)
        };

        loop {
            if start_time.elapsed() > timeout {
                return Err(QuantumError::TimeoutError {
                    seconds: timeout.as_secs(),
                });
            }

            let response = self
                .build_request(access_token, reqwest::Method::GET, &path)
                .send()
                .await?;

            if !response.status().is_success() {
                return Err(
                    Self::response_error("Failed to poll IBM job status", response, false).await,
                );
            }

            let job_data: Value = response.json().await?;
            let status = job_data["status"]
                .as_str()
                .unwrap_or("UNKNOWN")
                .to_uppercase();

            match status.as_str() {
                "COMPLETED" | "DONE" => {
                    let mut result = self.fetch_job_results(access_token, job_id).await?;
                    if result.get("id").is_none() {
                        result["id"] = Value::String(job_id.to_string());
                    }
                    if result.get("status").is_none() {
                        result["status"] = Value::String(status);
                    }
                    return Ok(result);
                }
                "CANCELLED" | "ERROR" | "FAILED" => {
                    return Err(QuantumError::BackendError {
                        message: format!("IBM job {} failed with status {}", job_id, status),
                    });
                }
                "RUNNING" | "QUEUED" | "PENDING" | "VALIDATING" => {
                    debug!("IBM job {} status: {}", job_id, status);
                }
                _ => {
                    warn!("IBM job {} returned unknown status {}", job_id, status);
                }
            }

            sleep(check_interval).await;
        }
    }

    fn circuit_to_qasm(&self, circuit: &QuantumCircuit) -> QuantumResult2<String> {
        let mut qasm = String::new();
        qasm.push_str("OPENQASM 3.0;\n");
        qasm.push_str("include \"stdgates.inc\";\n\n");
        qasm.push_str(&format!("qubit[{}] q;\n", circuit.qubits));
        qasm.push_str(&format!("bit[{}] c;\n", circuit.measurements.len()));

        for gate in &circuit.gates {
            match gate {
                QuantumGate::H(qubit) => qasm.push_str(&format!("h q[{}];\n", qubit)),
                QuantumGate::X(qubit) => qasm.push_str(&format!("x q[{}];\n", qubit)),
                QuantumGate::Y(qubit) => qasm.push_str(&format!("y q[{}];\n", qubit)),
                QuantumGate::Z(qubit) => qasm.push_str(&format!("z q[{}];\n", qubit)),
                QuantumGate::RX(qubit, angle) => {
                    qasm.push_str(&format!("rx({}) q[{}];\n", angle, qubit))
                }
                QuantumGate::RY(qubit, angle) => {
                    qasm.push_str(&format!("ry({}) q[{}];\n", angle, qubit))
                }
                QuantumGate::RZ(qubit, angle) => {
                    qasm.push_str(&format!("rz({}) q[{}];\n", angle, qubit))
                }
                QuantumGate::CNOT(control, target) => {
                    qasm.push_str(&format!("cx q[{}], q[{}];\n", control, target))
                }
                QuantumGate::CZ(control, target) => {
                    qasm.push_str(&format!("cz q[{}], q[{}];\n", control, target))
                }
                QuantumGate::CCNOT(control1, control2, target) => qasm.push_str(&format!(
                    "ccx q[{}], q[{}], q[{}];\n",
                    control1, control2, target
                )),
                QuantumGate::SacredFrequency(qubit, frequency) => {
                    let angle = frequency_to_quantum_angle(*frequency);
                    qasm.push_str(&format!("// Sacred frequency {} Hz\n", frequency));
                    qasm.push_str(&format!("ry({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::PhiHarmonic(qubit, phi_power) => {
                    let angle = phi_power_to_angle(*phi_power);
                    qasm.push_str(&format!("// Phi-harmonic φ^{}\n", phi_power));
                    qasm.push_str(&format!("rz({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::Custom(name, qubits, params) => {
                    qasm.push_str(&format!("// Custom gate: {}\n", name));
                    if !qubits.is_empty() && !params.is_empty() {
                        qasm.push_str(&format!("ry({}) q[{}];\n", params[0], qubits[0]));
                    }
                }
            }
        }

        for (i, &qubit) in circuit.measurements.iter().enumerate() {
            qasm.push_str(&format!("c[{}] = measure q[{}];\n", i, qubit));
        }

        Ok(qasm)
    }

    fn merge_counts_from_value(value: &Value, counts: &mut HashMap<String, u32>) -> bool {
        match value {
            Value::Object(map) => {
                if let Some(counts_value) = map.get("counts").and_then(Value::as_object) {
                    for (bitstring, count) in counts_value {
                        if let Some(count) = count.as_u64() {
                            counts.insert(bitstring.clone(), count as u32);
                        }
                    }
                    return !counts.is_empty();
                }

                if let Some(samples) = map.get("samples").and_then(Value::as_array) {
                    for sample in samples {
                        let key = sample
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| sample.to_string());
                        *counts.entry(key).or_insert(0) += 1;
                    }
                    return !counts.is_empty();
                }

                let mut found = false;
                for nested in map.values() {
                    found |= Self::merge_counts_from_value(nested, counts);
                }
                found
            }
            Value::Array(items) => {
                let mut found = false;
                for item in items {
                    found |= Self::merge_counts_from_value(item, counts);
                }
                found
            }
            _ => false,
        }
    }

    fn parse_job_result(&self, job_data: Value) -> QuantumResult2<QuantumResult> {
        let job_id = job_data["id"]
            .as_str()
            .or_else(|| job_data["job_id"].as_str())
            .unwrap_or("unknown")
            .to_string();
        let status = job_data["status"]
            .as_str()
            .unwrap_or("completed")
            .to_string();

        let mut counts = HashMap::new();
        Self::merge_counts_from_value(&job_data, &mut counts);

        Ok(QuantumResult {
            job_id,
            status,
            counts,
            statevector: None,
            execution_time: 0.0,
            backend_name: self.backend_name.clone(),
            metadata: HashMap::new(),
        })
    }

    pub async fn execute_openqasm(
        &self,
        openqasm: &str,
        shots: Option<u32>,
        timeout: Option<Duration>,
    ) -> QuantumResult2<QuantumResult> {
        let access_token = self.authenticate().await?;
        let job_id = self
            .submit_openqasm_job(&access_token, openqasm, shots.unwrap_or(self.shots))
            .await?;
        let job_result = self
            .wait_for_job(
                &access_token,
                &job_id,
                timeout.unwrap_or_else(|| Duration::from_secs(self.timeout_seconds)),
            )
            .await?;

        self.parse_job_result(job_result)
    }
}

#[async_trait]
impl QuantumBackend for IBMQuantumBackend {
    async fn initialize(&mut self, config: QuantumConfig) -> Result<(), QuantumError> {
        info!(
            "🔧 Initializing IBM Quantum backend: {}",
            config.backend_name
        );

        self.api_token = config.api_token.clone();
        self.service_crn = config.service_crn.clone();
        self.region = config.region.clone();
        self.hub = config.hub.clone();
        self.group = config.group.clone();
        self.project = config.project.clone();
        self.backend_name = config.backend_name.clone();
        self.shots = config.shots;
        self.timeout_seconds = config.timeout_seconds;
        self.base_url = if self.is_runtime_api() {
            Self::runtime_base_url(self.region.as_deref())
        } else {
            LEGACY_BASE_URL.to_string()
        };

        // Test authentication
        let _access_token = self.authenticate().await?;

        // Get backend capabilities
        let backend_info = self.get_backend_info(&_access_token).await?;

        let max_qubits = self.extract_max_qubits(&backend_info);
        let basis_gates = self.extract_basis_gates(&backend_info);

        self.capabilities = Some(QuantumCapabilities {
            max_qubits,
            gate_set: basis_gates.clone(),
            supports_sacred_frequencies: true, // Through rotation gates
            supports_phi_harmonic: true,       // Through rotation gates
            coupling_map: None,                // Would parse from backend_info
            basis_gates,
        });

        info!(
            "✅ IBM Quantum backend initialized - {} qubits available",
            max_qubits
        );
        Ok(())
    }

    async fn execute_circuit(
        &self,
        circuit: QuantumCircuit,
    ) -> Result<QuantumResult, QuantumError> {
        info!("🚀 Executing circuit on IBM Quantum backend");
        let qasm = self.circuit_to_qasm(&circuit)?;
        self.execute_openqasm(
            &qasm,
            Some(self.shots),
            Some(Duration::from_secs(self.timeout_seconds)),
        )
        .await
    }

    fn get_capabilities(&self) -> QuantumCapabilities {
        self.capabilities.clone().unwrap_or(QuantumCapabilities {
            max_qubits: 5,
            gate_set: vec!["h".to_string(), "x".to_string(), "cx".to_string()],
            supports_sacred_frequencies: true,
            supports_phi_harmonic: true,
            coupling_map: None,
            basis_gates: vec![
                "u1".to_string(),
                "u2".to_string(),
                "u3".to_string(),
                "cx".to_string(),
            ],
        })
    }

    async fn is_available(&self) -> bool {
        self.authenticate().await.is_ok()
    }

    async fn get_status(&self) -> Result<BackendStatus, QuantumError> {
        let access_token = self.authenticate().await?;
        let backend_info = self.get_backend_info(&access_token).await?;
        Ok(self.extract_backend_status(&backend_info))
    }

    async fn execute_sacred_frequency_operation(
        &self,
        frequency: u32,
        qubits: u32,
    ) -> Result<QuantumResult, QuantumError> {
        info!(
            "🎵 Executing sacred frequency {} Hz operation on {} qubits",
            frequency, qubits
        );

        if !is_sacred_frequency(frequency) {
            return Err(QuantumError::UnsupportedSacredFrequency { frequency });
        }

        // Create circuit with sacred frequency gates
        let mut gates = vec![];
        for qubit in 0..qubits {
            gates.push(QuantumGate::SacredFrequency(qubit, frequency));
        }

        let circuit = QuantumCircuit {
            qubits,
            gates,
            measurements: (0..qubits).collect(),
            metadata: [("sacred_frequency".to_string(), json!(frequency))]
                .iter()
                .cloned()
                .collect(),
        };

        self.execute_circuit(circuit).await
    }

    async fn execute_phi_gate(
        &self,
        qubit: u32,
        phi_power: f64,
    ) -> Result<QuantumResult, QuantumError> {
        info!(
            "🌀 Executing phi-harmonic gate φ^{} on qubit {}",
            phi_power, qubit
        );

        let circuit = QuantumCircuit {
            qubits: qubit + 1,
            gates: vec![QuantumGate::PhiHarmonic(qubit, phi_power)],
            measurements: vec![qubit],
            metadata: [("phi_power".to_string(), json!(phi_power))]
                .iter()
                .cloned()
                .collect(),
        };

        self.execute_circuit(circuit).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_runtime_sampler_fixture_recovers_counts() {
        let payload: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/ibm_runtime_sampler_result.json"
        ))
        .expect("fixture must parse");

        let backend = IBMQuantumBackend::with_backend("ibm_osaka".to_string());
        let result = backend
            .parse_job_result(payload)
            .expect("runtime payload should parse");

        assert_eq!(result.job_id, "job_123");
        assert_eq!(result.status, "COMPLETED");
        assert_eq!(result.counts.get("0x0"), Some(&3));
        assert_eq!(result.counts.get("0x1"), Some(&1));
    }
}
