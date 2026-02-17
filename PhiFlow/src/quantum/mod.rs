// PhiFlow Quantum Backends - Real quantum computer integration
// Supports IBM Quantum, Google Cirq, and other quantum providers

pub mod backends;
pub mod ibm_quantum;
pub mod simulator;
pub mod types;

pub use types::*;
pub use backends::*;
pub use simulator::QuantumSimulator;

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

#[async_trait]
pub trait QuantumBackend: Send + Sync {
    /// Initialize connection to quantum backend
    async fn initialize(&mut self, config: QuantumConfig) -> Result<(), QuantumError>;
    
    /// Execute quantum circuit on backend
    async fn execute_circuit(&self, circuit: QuantumCircuit) -> Result<QuantumResult, QuantumError>;
    
    /// Get backend capabilities (max qubits, etc.)
    fn get_capabilities(&self) -> QuantumCapabilities;
    
    /// Check if backend is available
    async fn is_available(&self) -> bool;
    
    /// Get backend status
    async fn get_status(&self) -> Result<BackendStatus, QuantumError>;
    
    /// Execute sacred frequency quantum operation
    async fn execute_sacred_frequency_operation(
        &self, 
        frequency: u32, 
        qubits: u32
    ) -> Result<QuantumResult, QuantumError>;
    
    /// Execute phi-harmonic gate
    async fn execute_phi_gate(
        &self,
        qubit: u32,
        phi_power: f64
    ) -> Result<QuantumResult, QuantumError>;
}

#[derive(Debug, Clone)]
pub struct QuantumConfig {
    pub backend_name: String,
    pub api_token: Option<String>,
    pub hub: Option<String>,
    pub group: Option<String>,
    pub project: Option<String>,
    pub max_qubits: u32,
    pub shots: u32,
    pub timeout_seconds: u64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        QuantumConfig {
            backend_name: "simulator".to_string(),
            api_token: None,
            hub: None,
            group: None,
            project: None,
            max_qubits: 32,
            shots: 1024,
            timeout_seconds: 300,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub qubits: u32,
    pub gates: Vec<QuantumGate>,
    pub measurements: Vec<u32>,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub enum QuantumGate {
    H(u32),                                    // Hadamard
    X(u32),                                    // Pauli-X
    Y(u32),                                    // Pauli-Y
    Z(u32),                                    // Pauli-Z
    RX(u32, f64),                             // X rotation
    RY(u32, f64),                             // Y rotation
    RZ(u32, f64),                             // Z rotation
    CNOT(u32, u32),                           // Controlled-NOT
    CZ(u32, u32),                             // Controlled-Z
    CCNOT(u32, u32, u32),                     // Toffoli
    SacredFrequency(u32, u32),                // Sacred frequency gate
    PhiHarmonic(u32, f64),                    // Phi-harmonic gate
    Custom(String, Vec<u32>, Vec<f64>),       // Custom gate
}

#[derive(Debug, Clone)]
pub struct QuantumResult {
    pub job_id: String,
    pub status: String,
    pub counts: HashMap<String, u32>,
    pub statevector: Option<Vec<num_complex::Complex64>>,
    pub execution_time: f64,
    pub backend_name: String,
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct QuantumCapabilities {
    pub max_qubits: u32,
    pub gate_set: Vec<String>,
    pub supports_sacred_frequencies: bool,
    pub supports_phi_harmonic: bool,
    pub coupling_map: Option<Vec<(u32, u32)>>,
    pub basis_gates: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub operational: bool,
    pub pending_jobs: u32,
    pub queue_length: u32,
    pub status_msg: String,
    pub last_update: String,
}

#[derive(Debug, thiserror::Error)]
pub enum QuantumError {
    #[error("Backend error: {message}")]
    BackendError { message: String },
    
    #[error("Authentication error: {message}")]
    AuthError { message: String },
    
    #[error("Circuit error: {message}")]
    CircuitError { message: String },
    
    #[error("Network error: {source}")]
    NetworkError { 
        #[from] 
        source: reqwest::Error 
    },
    
    #[error("Serialization error: {source}")]
    SerializationError { 
        #[from] 
        source: serde_json::Error 
    },
    
    #[error("Timeout error: operation took longer than {seconds} seconds")]
    TimeoutError { seconds: u64 },
    
    #[error("Sacred frequency {frequency} not supported by backend")]
    UnsupportedSacredFrequency { frequency: u32 },
    
    #[error("Phi-harmonic operation {phi_power} not supported by backend")]
    UnsupportedPhiHarmonic { phi_power: f64 },
}

pub type QuantumResult2<T> = Result<T, QuantumError>;