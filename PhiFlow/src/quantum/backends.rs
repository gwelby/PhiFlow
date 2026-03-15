// PhiFlow Quantum Backend Manager - Manages multiple quantum backends

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

use super::types::*;
use super::simulator::QuantumSimulator;
use super::ibm_quantum::IBMQuantumBackend;

pub struct QuantumBackendManager {
    backends: HashMap<String, Arc<Mutex<dyn QuantumBackend>>>,
    default_backend: String,
    config: QuantumConfig,
}

impl QuantumBackendManager {
    pub fn new() -> Self {
        QuantumBackendManager {
            backends: HashMap::new(),
            default_backend: "simulator".to_string(),
            config: QuantumConfig::default(),
        }
    }

    pub async fn initialize(&mut self, config: QuantumConfig) -> Result<(), QuantumError> {
        info!("🔧 Initializing quantum backend manager");
        
        self.config = config.clone();
        
        // Always register the simulator
        let simulator = Arc::new(Mutex::new(QuantumSimulator::with_max_qubits(config.max_qubits)));
        simulator.lock().await.initialize(config.clone()).await?;
        self.backends.insert("simulator".to_string(), simulator);
        info!("✅ Quantum simulator registered");

        // Register IBM Quantum if token is provided
        if config.api_token.is_some() {
            let ibm_backend = Arc::new(Mutex::new(IBMQuantumBackend::with_backend(config.backend_name.clone())));
            let init_result = {
                let mut backend = ibm_backend.lock().await;
                backend.initialize(config.clone()).await
            };
            match init_result {
                Ok(()) => {
                    self.backends.insert("ibm".to_string(), ibm_backend);
                    info!("✅ IBM Quantum backend registered");
                }
                Err(e) => {
                    warn!("⚠️ Failed to initialize IBM Quantum backend: {}", e);
                }
            }
        }

        // Set default backend
        if self.backends.contains_key(&config.backend_name) {
            self.default_backend = config.backend_name.clone();
        } else if config.backend_name.starts_with("ibmq_") && self.backends.contains_key("ibm") {
            self.default_backend = "ibm".to_string();
        }

        info!("🎯 Default quantum backend: {}", self.default_backend);
        Ok(())
    }

    pub async fn execute_circuit(&self, circuit: QuantumCircuit, backend_name: Option<&str>) -> Result<QuantumResult, QuantumError> {
        let backend_key = backend_name.unwrap_or(&self.default_backend);
        
        let backend = self.backends.get(backend_key)
            .ok_or_else(|| QuantumError::BackendError { 
                message: format!("Backend '{}' not found", backend_key) 
            })?;

        backend.lock().await.execute_circuit(circuit).await
    }

    pub async fn execute_sacred_frequency_operation(
        &self, 
        frequency: u32, 
        qubits: u32, 
        backend_name: Option<&str>
    ) -> Result<QuantumResult, QuantumError> {
        let backend_key = backend_name.unwrap_or(&self.default_backend);
        
        let backend = self.backends.get(backend_key)
            .ok_or_else(|| QuantumError::BackendError { 
                message: format!("Backend '{}' not found", backend_key) 
            })?;

        backend.lock().await.execute_sacred_frequency_operation(frequency, qubits).await
    }

    pub async fn execute_phi_gate(
        &self, 
        qubit: u32, 
        phi_power: f64, 
        backend_name: Option<&str>
    ) -> Result<QuantumResult, QuantumError> {
        let backend_key = backend_name.unwrap_or(&self.default_backend);
        
        let backend = self.backends.get(backend_key)
            .ok_or_else(|| QuantumError::BackendError { 
                message: format!("Backend '{}' not found", backend_key) 
            })?;

        backend.lock().await.execute_phi_gate(qubit, phi_power).await
    }

    pub async fn get_backend_capabilities(&self, backend_name: Option<&str>) -> Result<QuantumCapabilities, QuantumError> {
        let backend_key = backend_name.unwrap_or(&self.default_backend);
        
        let backend = self.backends.get(backend_key)
            .ok_or_else(|| QuantumError::BackendError { 
                message: format!("Backend '{}' not found", backend_key) 
            })?;

        Ok(backend.lock().await.get_capabilities())
    }

    pub async fn get_backend_status(&self, backend_name: Option<&str>) -> Result<BackendStatus, QuantumError> {
        let backend_key = backend_name.unwrap_or(&self.default_backend);
        
        let backend = self.backends.get(backend_key)
            .ok_or_else(|| QuantumError::BackendError { 
                message: format!("Backend '{}' not found", backend_key) 
            })?;

        backend.lock().await.get_status().await
    }

    pub async fn list_backends(&self) -> Vec<(String, bool)> {
        let mut result = Vec::new();
        
        for (name, backend) in &self.backends {
            let available = backend.lock().await.is_available().await;
            result.push((name.clone(), available));
        }
        
        result
    }

    pub fn get_default_backend(&self) -> &str {
        &self.default_backend
    }

    pub fn set_default_backend(&mut self, backend_name: String) -> Result<(), QuantumError> {
        if self.backends.contains_key(&backend_name) {
            self.default_backend = backend_name;
            info!("🎯 Default backend changed to: {}", self.default_backend);
            Ok(())
        } else {
            Err(QuantumError::BackendError { 
                message: format!("Backend '{}' not found", backend_name) 
            })
        }
    }

    pub async fn test_all_backends(&self) -> HashMap<String, Result<(), QuantumError>> {
        let mut results = HashMap::new();
        
        for (name, backend) in &self.backends {
            info!("🧪 Testing backend: {}", name);
            
            // Create simple test circuit
            let test_circuit = QuantumCircuit {
                qubits: 1,
                gates: vec![QuantumGate::H(0)],
                measurements: vec![0],
                metadata: HashMap::new(),
            };
            
            let result = match backend.lock().await.execute_circuit(test_circuit).await {
                Ok(_) => {
                    info!("✅ Backend {} test passed", name);
                    Ok(())
                }
                Err(e) => {
                    error!("❌ Backend {} test failed: {}", name, e);
                    Err(e)
                }
            };
            
            results.insert(name.clone(), result);
        }
        
        results
    }

    pub async fn get_sacred_frequency_support(&self) -> HashMap<String, bool> {
        let mut support = HashMap::new();
        
        for (name, backend) in &self.backends {
            let capabilities = backend.lock().await.get_capabilities();
            support.insert(name.clone(), capabilities.supports_sacred_frequencies);
        }
        
        support
    }

    pub async fn get_phi_harmonic_support(&self) -> HashMap<String, bool> {
        let mut support = HashMap::new();
        
        for (name, backend) in &self.backends {
            let capabilities = backend.lock().await.get_capabilities();
            support.insert(name.clone(), capabilities.supports_phi_harmonic);
        }
        
        support
    }

    pub async fn execute_consciousness_coupled_circuit(
        &self,
        circuit: QuantumCircuit,
        consciousness_coupling: ConsciousnessQuantumCoupling,
        backend_name: Option<&str>
    ) -> Result<QuantumResult, QuantumError> {
        if !consciousness_coupling.quantum_authorization {
            return Err(QuantumError::BackendError {
                message: "Quantum operations not authorized by consciousness monitor".to_string()
            });
        }

        info!("🧠 Executing consciousness-coupled quantum circuit");
        info!("   Coherence: {:.3}", consciousness_coupling.coherence_threshold);
        if let Some(freq) = consciousness_coupling.sacred_frequency_lock {
            info!("   Sacred frequency lock: {} Hz", freq);
        }
        info!("   Phi resonance: {:.3}", consciousness_coupling.phi_resonance);

        self.execute_circuit(circuit, backend_name).await
    }
}

impl Default for QuantumBackendManager {
    fn default() -> Self {
        Self::new()
    }
}

// Factory functions for creating specific backends
pub fn create_simulator_backend(max_qubits: u32) -> Box<dyn QuantumBackend> {
    Box::new(QuantumSimulator::with_max_qubits(max_qubits))
}

pub fn create_ibm_backend(backend_name: String) -> Box<dyn QuantumBackend> {
    Box::new(IBMQuantumBackend::with_backend(backend_name))
}

// Utility function to detect available backends
pub async fn detect_available_backends(config: &QuantumConfig) -> Vec<String> {
    let mut available = vec!["simulator".to_string()];
    
    // Test IBM Quantum connectivity
    if config.api_token.is_some() {
        let mut ibm_backend = IBMQuantumBackend::new();
        if ibm_backend.initialize(config.clone()).await.is_ok() {
            available.push("ibm".to_string());
        }
    }
    
    available
}