// PhiFlow IBM Quantum Backend - Real IBM Quantum computer integration
// Provides connectivity to IBM Quantum Experience via REST API

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{info, warn, error, debug};

use super::types::*;

pub struct IBMQuantumBackend {
    client: Client,
    api_token: Option<String>,
    base_url: String,
    hub: Option<String>,
    group: Option<String>,
    project: Option<String>,
    backend_name: String,
    capabilities: Option<QuantumCapabilities>,
}

impl IBMQuantumBackend {
    pub fn new() -> Self {
        IBMQuantumBackend {
            client: Client::new(),
            api_token: None,
            base_url: "https://api.quantum-computing.ibm.com".to_string(),
            hub: None,
            group: None,
            project: None,
            backend_name: "ibmq_qasm_simulator".to_string(),
            capabilities: None,
        }
    }

    pub fn with_backend(backend_name: String) -> Self {
        let mut backend = Self::new();
        backend.backend_name = backend_name;
        backend
    }

    async fn authenticate(&self) -> QuantumResult2<String> {
        let token = self.api_token
            .as_ref()
            .ok_or_else(|| QuantumError::AuthError { 
                message: "No API token provided".to_string() 
            })?;

        let response = self.client
            .post(&format!("{}/api/Network", self.base_url))
            .header("X-Qx-Token", token)
            .json(&json!({
                "apiToken": token
            }))
            .send()
            .await?;

        if response.status().is_success() {
            let data: Value = response.json().await?;
            info!("✅ IBM Quantum authentication successful");
            Ok(data["accessToken"].as_str().unwrap_or("").to_string())
        } else {
            Err(QuantumError::AuthError {
                message: format!("Authentication failed: {}", response.status())
            })
        }
    }

    async fn get_backend_info(&self, access_token: &str) -> QuantumResult2<Value> {
        let url = format!("{}/api/Backends/{}", self.base_url, self.backend_name);
        
        let response = self.client
            .get(&url)
            .header("X-Access-Token", access_token)
            .send()
            .await?;

        if response.status().is_success() {
            Ok(response.json().await?)
        } else {
            Err(QuantumError::BackendError {
                message: format!("Failed to get backend info: {}", response.status())
            })
        }
    }

    async fn submit_job(&self, access_token: &str, circuit: &QuantumCircuit) -> QuantumResult2<String> {
        // Convert PhiFlow circuit to QASM
        let qasm = self.circuit_to_qasm(circuit)?;
        
        info!("🔧 Converting PhiFlow circuit to QASM:");
        debug!("QASM circuit:\n{}", qasm);

        let job_data = json!({
            "qObject": {
                "qobj_id": uuid::Uuid::new_v4().to_string(),
                "type": "QASM",
                "schema_version": "1.3.0",
                "experiments": [{
                    "name": "phiflow_experiment",
                    "qasm": qasm
                }],
                "backend_name": self.backend_name,
                "shots": 1024
            }
        });

        let url = format!("{}/api/Jobs", self.base_url);
        
        let response = self.client
            .post(&url)
            .header("X-Access-Token", access_token)
            .header("Content-Type", "application/json")
            .json(&job_data)
            .send()
            .await?;

        if response.status().is_success() {
            let result: Value = response.json().await?;
            let job_id = result["id"].as_str()
                .ok_or_else(|| QuantumError::BackendError { 
                    message: "No job ID in response".to_string() 
                })?;
            
            info!("✅ Job submitted to IBM Quantum: {}", job_id);
            Ok(job_id.to_string())
        } else {
            Err(QuantumError::BackendError {
                message: format!("Job submission failed: {}", response.status())
            })
        }
    }

    async fn wait_for_job(&self, access_token: &str, job_id: &str, timeout: Duration) -> QuantumResult2<Value> {
        let start_time = Instant::now();
        let check_interval = Duration::from_secs(2);

        loop {
            if start_time.elapsed() > timeout {
                return Err(QuantumError::TimeoutError { 
                    seconds: timeout.as_secs() 
                });
            }

            let url = format!("{}/api/Jobs/{}", self.base_url, job_id);
            let response = self.client
                .get(&url)
                .header("X-Access-Token", access_token)
                .send()
                .await?;

            if response.status().is_success() {
                let job_data: Value = response.json().await?;
                let status = job_data["status"].as_str().unwrap_or("UNKNOWN");

                match status {
                    "COMPLETED" => {
                        info!("✅ IBM Quantum job completed: {}", job_id);
                        return Ok(job_data);
                    }
                    "CANCELLED" | "ERROR" => {
                        return Err(QuantumError::BackendError {
                            message: format!("Job failed with status: {}", status)
                        });
                    }
                    "RUNNING" | "QUEUED" => {
                        debug!("🔄 Job {} status: {}", job_id, status);
                    }
                    _ => {
                        warn!("Unknown job status: {}", status);
                    }
                }
            }

            sleep(check_interval).await;
        }
    }

    fn circuit_to_qasm(&self, circuit: &QuantumCircuit) -> QuantumResult2<String> {
        let mut qasm = String::new();
        qasm.push_str("OPENQASM 2.0;\n");
        qasm.push_str("include \"qelib1.inc\";\n");
        qasm.push_str(&format!("qreg q[{}];\n", circuit.qubits));
        qasm.push_str(&format!("creg c[{}];\n", circuit.measurements.len()));

        for gate in &circuit.gates {
            match gate {
                QuantumGate::H(qubit) => {
                    qasm.push_str(&format!("h q[{}];\n", qubit));
                }
                QuantumGate::X(qubit) => {
                    qasm.push_str(&format!("x q[{}];\n", qubit));
                }
                QuantumGate::Y(qubit) => {
                    qasm.push_str(&format!("y q[{}];\n", qubit));
                }
                QuantumGate::Z(qubit) => {
                    qasm.push_str(&format!("z q[{}];\n", qubit));
                }
                QuantumGate::RX(qubit, angle) => {
                    qasm.push_str(&format!("rx({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::RY(qubit, angle) => {
                    qasm.push_str(&format!("ry({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::RZ(qubit, angle) => {
                    qasm.push_str(&format!("rz({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::CNOT(control, target) => {
                    qasm.push_str(&format!("cx q[{}],q[{}];\n", control, target));
                }
                QuantumGate::CZ(control, target) => {
                    qasm.push_str(&format!("cz q[{}],q[{}];\n", control, target));
                }
                QuantumGate::CCNOT(control1, control2, target) => {
                    qasm.push_str(&format!("ccx q[{}],q[{}],q[{}];\n", control1, control2, target));
                }
                QuantumGate::SacredFrequency(qubit, frequency) => {
                    // Convert sacred frequency to quantum rotation
                    let angle = frequency_to_quantum_angle(*frequency);
                    qasm.push_str(&format!("// Sacred frequency {} Hz\n", frequency));
                    qasm.push_str(&format!("ry({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::PhiHarmonic(qubit, phi_power) => {
                    // Convert phi-harmonic to quantum rotation
                    let angle = phi_power_to_angle(*phi_power);
                    qasm.push_str(&format!("// Phi-harmonic φ^{}\n", phi_power));
                    qasm.push_str(&format!("rz({}) q[{}];\n", angle, qubit));
                }
                QuantumGate::Custom(name, qubits, params) => {
                    qasm.push_str(&format!("// Custom gate: {}\n", name));
                    // For now, convert to basic rotation
                    if !qubits.is_empty() && !params.is_empty() {
                        qasm.push_str(&format!("ry({}) q[{}];\n", params[0], qubits[0]));
                    }
                }
            }
        }

        // Add measurements
        for (i, &qubit) in circuit.measurements.iter().enumerate() {
            qasm.push_str(&format!("measure q[{}] -> c[{}];\n", qubit, i));
        }

        Ok(qasm)
    }

    fn parse_job_result(&self, job_data: Value) -> QuantumResult2<QuantumResult> {
        let job_id = job_data["id"].as_str().unwrap_or("unknown").to_string();
        let status = job_data["status"].as_str().unwrap_or("unknown").to_string();

        let mut counts = HashMap::new();
        
        if let Some(result) = job_data["qObjectResult"].as_object() {
            if let Some(results) = result["results"].as_array() {
                if let Some(experiment) = results.first() {
                    if let Some(data) = experiment["data"].as_object() {
                        if let Some(counts_obj) = data["counts"].as_object() {
                            for (bitstring, count) in counts_obj {
                                if let Some(count_val) = count.as_u64() {
                                    counts.insert(bitstring.clone(), count_val as u32);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(QuantumResult {
            job_id,
            status,
            counts,
            statevector: None, // IBM doesn't typically return statevector for real devices
            execution_time: 0.0, // Would need to calculate from timestamps
            backend_name: self.backend_name.clone(),
            metadata: HashMap::new(),
        })
    }
}

#[async_trait]
impl QuantumBackend for IBMQuantumBackend {
    async fn initialize(&mut self, config: QuantumConfig) -> Result<(), QuantumError> {
        info!("🔧 Initializing IBM Quantum backend: {}", config.backend_name);
        
        self.api_token = config.api_token.clone();
        self.hub = config.hub.clone();
        self.group = config.group.clone();
        self.project = config.project.clone();
        self.backend_name = config.backend_name.clone();

        // Test authentication
        let _access_token = self.authenticate().await?;
        
        // Get backend capabilities
        let backend_info = self.get_backend_info(&_access_token).await?;
        
        let max_qubits = backend_info["nQubits"].as_u64().unwrap_or(5) as u32;
        let basis_gates: Vec<String> = backend_info["basisGates"]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        self.capabilities = Some(QuantumCapabilities {
            max_qubits,
            gate_set: basis_gates.clone(),
            supports_sacred_frequencies: true,  // Through rotation gates
            supports_phi_harmonic: true,        // Through rotation gates
            coupling_map: None,                 // Would parse from backend_info
            basis_gates,
        });

        info!("✅ IBM Quantum backend initialized - {} qubits available", max_qubits);
        Ok(())
    }

    async fn execute_circuit(&self, circuit: QuantumCircuit) -> Result<QuantumResult, QuantumError> {
        info!("🚀 Executing circuit on IBM Quantum backend");
        
        let access_token = self.authenticate().await?;
        let job_id = self.submit_job(&access_token, &circuit).await?;
        let job_result = self.wait_for_job(&access_token, &job_id, Duration::from_secs(300)).await?;
        
        self.parse_job_result(job_result)
    }

    fn get_capabilities(&self) -> QuantumCapabilities {
        self.capabilities.clone().unwrap_or(QuantumCapabilities {
            max_qubits: 5,
            gate_set: vec!["h".to_string(), "x".to_string(), "cx".to_string()],
            supports_sacred_frequencies: true,
            supports_phi_harmonic: true,
            coupling_map: None,
            basis_gates: vec!["u1".to_string(), "u2".to_string(), "u3".to_string(), "cx".to_string()],
        })
    }

    async fn is_available(&self) -> bool {
        self.authenticate().await.is_ok()
    }

    async fn get_status(&self) -> Result<BackendStatus, QuantumError> {
        let access_token = self.authenticate().await?;
        let backend_info = self.get_backend_info(&access_token).await?;
        
        let operational = backend_info["status"].as_str() == Some("ONLINE");
        let pending_jobs = backend_info["lengthQueue"].as_u64().unwrap_or(0) as u32;
        
        Ok(BackendStatus {
            operational,
            pending_jobs,
            queue_length: pending_jobs,
            status_msg: backend_info["statusMsg"].as_str().unwrap_or("Unknown").to_string(),
            last_update: chrono::Utc::now().to_rfc3339(),
        })
    }

    async fn execute_sacred_frequency_operation(&self, frequency: u32, qubits: u32) -> Result<QuantumResult, QuantumError> {
        info!("🎵 Executing sacred frequency {} Hz operation on {} qubits", frequency, qubits);
        
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
            metadata: [("sacred_frequency".to_string(), json!(frequency))].iter().cloned().collect(),
        };

        self.execute_circuit(circuit).await
    }

    async fn execute_phi_gate(&self, qubit: u32, phi_power: f64) -> Result<QuantumResult, QuantumError> {
        info!("🌀 Executing phi-harmonic gate φ^{} on qubit {}", phi_power, qubit);
        
        let circuit = QuantumCircuit {
            qubits: qubit + 1,
            gates: vec![QuantumGate::PhiHarmonic(qubit, phi_power)],
            measurements: vec![qubit],
            metadata: [("phi_power".to_string(), json!(phi_power))].iter().cloned().collect(),
        };

        self.execute_circuit(circuit).await
    }
}