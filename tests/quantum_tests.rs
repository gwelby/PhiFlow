// PhiFlow Quantum Tests - Specialized tests for quantum computing functionality

use phiflow::quantum::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_quantum_simulator_basic_gates() {
    use phiflow::quantum::simulator::QuantumSimulator;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut simulator = QuantumSimulator::new();
    simulator.initialize(config).await.expect("Simulator initialization should succeed");
    
    // Test Hadamard gate
    let hadamard_circuit = QuantumCircuit {
        qubits: 1,
        gates: vec![QuantumGate::H(0)],
        measurements: vec![0],
        metadata: HashMap::new(),
    };
    
    let result = simulator.execute_circuit(hadamard_circuit).await.expect("Hadamard execution should succeed");
    assert_eq!(result.status, "COMPLETED");
    assert!(!result.counts.is_empty());
    
    // Test Pauli-X gate
    let x_circuit = QuantumCircuit {
        qubits: 1,
        gates: vec![QuantumGate::X(0)],
        measurements: vec![0],
        metadata: HashMap::new(),
    };
    
    let result = simulator.execute_circuit(x_circuit).await.expect("Pauli-X execution should succeed");
    assert_eq!(result.status, "COMPLETED");
    
    // With X gate, we should get |1⟩ state, so "1" should be most common
    assert!(result.counts.contains_key("1"));
}

#[tokio::test]
async fn test_quantum_simulator_cnot_gate() {
    use phiflow::quantum::simulator::QuantumSimulator;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut simulator = QuantumSimulator::new();
    simulator.initialize(config).await.expect("Simulator initialization should succeed");
    
    // Test Bell state preparation: H(0) -> CNOT(0,1)
    let bell_circuit = QuantumCircuit {
        qubits: 2,
        gates: vec![
            QuantumGate::H(0),
            QuantumGate::CNOT(0, 1),
        ],
        measurements: vec![0, 1],
        metadata: HashMap::new(),
    };
    
    let result = simulator.execute_circuit(bell_circuit).await.expect("Bell state execution should succeed");
    assert_eq!(result.status, "COMPLETED");
    assert!(!result.counts.is_empty());
    
    // Bell state should give roughly equal probabilities for "00" and "11"
    assert!(result.counts.contains_key("00") || result.counts.contains_key("11"));
}

#[tokio::test]
async fn test_quantum_simulator_sacred_frequency_gates() {
    use phiflow::quantum::simulator::QuantumSimulator;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut simulator = QuantumSimulator::new();
    simulator.initialize(config).await.expect("Simulator initialization should succeed");
    
    let sacred_frequencies = [432, 528, 594, 720, 963];
    
    for frequency in sacred_frequencies {
        let circuit = QuantumCircuit {
            qubits: 1,
            gates: vec![QuantumGate::SacredFrequency(0, frequency)],
            measurements: vec![0],
            metadata: HashMap::new(),
        };
        
        let result = simulator
            .execute_circuit(circuit)
            .await
            .unwrap_or_else(|_| panic!("Sacred frequency {} execution should succeed", frequency));
        
        assert_eq!(result.status, "COMPLETED");
        assert!(!result.counts.is_empty());
    }
}

#[tokio::test]
async fn test_quantum_simulator_phi_harmonic_gates() {
    use phiflow::quantum::simulator::QuantumSimulator;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut simulator = QuantumSimulator::new();
    simulator.initialize(config).await.expect("Simulator initialization should succeed");
    
    let phi_powers = [1.0, 1.618, 2.618, 4.236]; // φ^0, φ^1, φ^2, φ^3
    
    for phi_power in phi_powers {
        let circuit = QuantumCircuit {
            qubits: 1,
            gates: vec![QuantumGate::PhiHarmonic(0, phi_power)],
            measurements: vec![0],
            metadata: HashMap::new(),
        };
        
        let result = simulator
            .execute_circuit(circuit)
            .await
            .unwrap_or_else(|_| panic!("Phi-harmonic φ^{} execution should succeed", phi_power));
        
        assert_eq!(result.status, "COMPLETED");
        assert!(!result.counts.is_empty());
    }
}

#[tokio::test]
async fn test_quantum_simulator_rotation_gates() {
    use phiflow::quantum::simulator::QuantumSimulator;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut simulator = QuantumSimulator::new();
    simulator.initialize(config).await.expect("Simulator initialization should succeed");
    
    // Test RX gate
    let rx_circuit = QuantumCircuit {
        qubits: 1,
        gates: vec![QuantumGate::RX(0, std::f64::consts::PI / 2.0)],
        measurements: vec![0],
        metadata: HashMap::new(),
    };
    
    let result = simulator.execute_circuit(rx_circuit).await.expect("RX execution should succeed");
    assert_eq!(result.status, "COMPLETED");
    
    // Test RY gate
    let ry_circuit = QuantumCircuit {
        qubits: 1,
        gates: vec![QuantumGate::RY(0, std::f64::consts::PI / 4.0)],
        measurements: vec![0],
        metadata: HashMap::new(),
    };
    
    let result = simulator.execute_circuit(ry_circuit).await.expect("RY execution should succeed");
    assert_eq!(result.status, "COMPLETED");
    
    // Test RZ gate
    let rz_circuit = QuantumCircuit {
        qubits: 1,
        gates: vec![QuantumGate::RZ(0, std::f64::consts::PI / 3.0)],
        measurements: vec![0],
        metadata: HashMap::new(),
    };
    
    let result = simulator.execute_circuit(rz_circuit).await.expect("RZ execution should succeed");
    assert_eq!(result.status, "COMPLETED");
}

#[tokio::test]
async fn test_quantum_backend_manager_initialization() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 8,
        shots: 2048,
        timeout_seconds: 600,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let backends = backend_manager.list_backends().await;
    assert!(!backends.is_empty());
    assert!(backends.iter().any(|(name, available)| name == "simulator" && *available));
}

#[tokio::test]
async fn test_quantum_backend_manager_capabilities() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 16,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let capabilities = backend_manager.get_backend_capabilities(None).await
        .expect("Should get capabilities");
    
    assert!(capabilities.max_qubits >= 16);
    assert!(capabilities.supports_sacred_frequencies);
    assert!(capabilities.supports_phi_harmonic);
    assert!(!capabilities.gate_set.is_empty());
}

#[tokio::test]
async fn test_quantum_backend_manager_status() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let status = backend_manager.get_backend_status(None).await
        .expect("Should get status");
    
    assert!(status.operational);
    assert_eq!(status.pending_jobs, 0);
    assert_eq!(status.queue_length, 0);
    assert_eq!(status.status_msg, "Simulator ready");
}

#[tokio::test]
async fn test_quantum_backend_manager_sacred_frequency_support() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let sacred_support = backend_manager.get_sacred_frequency_support().await;
    assert!(!sacred_support.is_empty());
    assert_eq!(sacred_support.get("simulator"), Some(&true));
}

#[tokio::test]
async fn test_quantum_backend_manager_phi_harmonic_support() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let phi_support = backend_manager.get_phi_harmonic_support().await;
    assert!(!phi_support.is_empty());
    assert_eq!(phi_support.get("simulator"), Some(&true));
}

#[tokio::test]
async fn test_quantum_backend_manager_test_all() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let test_results = backend_manager.test_all_backends().await;
    assert!(!test_results.is_empty());
    assert!(test_results.get("simulator").unwrap().is_ok());
}

#[tokio::test]
async fn test_consciousness_quantum_coupling() {
    use phiflow::quantum::types::ConsciousnessQuantumCoupling;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    // Test with authorized consciousness coupling
    let authorized_coupling = ConsciousnessQuantumCoupling {
        coherence_threshold: 0.85,
        sacred_frequency_lock: Some(528),
        phi_resonance: 0.92,
        quantum_authorization: true,
    };
    
    let circuit = QuantumCircuit {
        qubits: 2,
        gates: vec![
            QuantumGate::H(0),
            QuantumGate::SacredFrequency(1, 528),
        ],
        measurements: vec![0, 1],
        metadata: HashMap::new(),
    };
    
    let result = backend_manager.execute_consciousness_coupled_circuit(
        circuit, 
        authorized_coupling, 
        Some("simulator")
    ).await.expect("Consciousness-coupled execution should succeed");
    
    assert_eq!(result.status, "COMPLETED");
    
    // Test with unauthorized consciousness coupling
    let unauthorized_coupling = ConsciousnessQuantumCoupling {
        coherence_threshold: 0.50,
        sacred_frequency_lock: None,
        phi_resonance: 0.30,
        quantum_authorization: false,
    };
    
    let circuit2 = QuantumCircuit {
        qubits: 1,
        gates: vec![QuantumGate::H(0)],
        measurements: vec![0],
        metadata: HashMap::new(),
    };
    
    let result = backend_manager.execute_consciousness_coupled_circuit(
        circuit2, 
        unauthorized_coupling, 
        Some("simulator")
    ).await;
    
    assert!(result.is_err()); // Should fail due to lack of authorization
}

#[tokio::test]
async fn test_complex_sacred_frequency_phi_circuit() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    // Create a complex circuit combining sacred frequencies and phi-harmonic gates
    let circuit = QuantumCircuit {
        qubits: 4,
        gates: vec![
            // Initialize with Hadamard
            QuantumGate::H(0),
            
            // Apply sacred frequencies to different qubits
            QuantumGate::SacredFrequency(1, 432), // Earth frequency
            QuantumGate::SacredFrequency(2, 528), // Love frequency
            QuantumGate::SacredFrequency(3, 963), // Unity consciousness
            
            // Apply phi-harmonic transformations
            QuantumGate::PhiHarmonic(0, 1.0),     // φ^0
            QuantumGate::PhiHarmonic(1, 1.618),   // φ^1
            QuantumGate::PhiHarmonic(2, 2.618),   // φ^2
            QuantumGate::PhiHarmonic(3, 4.236),   // φ^3
            
            // Create entanglement
            QuantumGate::CNOT(0, 1),
            QuantumGate::CNOT(1, 2),
            QuantumGate::CNOT(2, 3),
            
            // Additional sacred frequency modulation
            QuantumGate::SacredFrequency(0, 720), // Vision frequency
        ],
        measurements: vec![0, 1, 2, 3],
        metadata: [
            ("experiment_type".to_string(), serde_json::json!("consciousness_quantum_symphony")),
            ("sacred_frequencies".to_string(), serde_json::json!([432, 528, 720, 963])),
            ("phi_powers".to_string(), serde_json::json!([1.0, 1.618, 2.618, 4.236])),
            ("consciousness_level".to_string(), serde_json::json!("transcendent")),
        ].iter().cloned().collect(),
    };
    
    let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
        .expect("Complex consciousness-quantum circuit execution should succeed");
    
    assert_eq!(result.status, "COMPLETED");
    assert!(!result.counts.is_empty());
    assert!(result.metadata.contains_key("experiment_type"));
    assert!(result.metadata.contains_key("sacred_frequencies"));
    assert!(result.metadata.contains_key("phi_powers"));
    assert!(result.execution_time >= 0.0);
    
    // Should have 16 possible measurement outcomes for 4 qubits (2^4)
    let total_counts: u32 = result.counts.values().sum();
    assert!(total_counts > 0);
}

#[tokio::test]
async fn test_quantum_error_recovery() {
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 2, // Deliberately small for testing limits
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    // Test circuit that exceeds qubit limit
    let too_large_circuit = QuantumCircuit {
        qubits: 10, // Exceeds the 2-qubit limit
        gates: vec![QuantumGate::H(0), QuantumGate::H(5)],
        measurements: vec![0, 5],
        metadata: HashMap::new(),
    };
    
    let result = backend_manager.execute_circuit(too_large_circuit, Some("simulator")).await;
    assert!(result.is_err()); // Should fail due to qubit limit
    
    // Test invalid sacred frequency
    let result = backend_manager.execute_sacred_frequency_operation(999, 1, Some("simulator")).await;
    assert!(result.is_err()); // Should fail due to invalid frequency
    
    // Test valid circuit within limits
    let valid_circuit = QuantumCircuit {
        qubits: 2,
        gates: vec![QuantumGate::H(0), QuantumGate::CNOT(0, 1)],
        measurements: vec![0, 1],
        metadata: HashMap::new(),
    };
    
    let result = backend_manager.execute_circuit(valid_circuit, Some("simulator")).await;
    assert!(result.is_ok()); // Should succeed
}
