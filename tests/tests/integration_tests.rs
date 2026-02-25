// PhiFlow Integration Tests - Comprehensive testing for the complete PhiFlow system
// Tests the full pipeline from source code to quantum execution

use phiflow::*;
use tokio;
use std::collections::HashMap;

#[tokio::test]
async fn test_basic_phiflow_program() {
    let source = r#"
        let x = 42;
        let y = PHI * 2;
        x + y
    "#;
    
    let mut lexer = PhiFlowLexer::new(source.to_string());
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    let mut parser = PhiFlowParser::new(tokens);
    let program = parser.parse().expect("Parsing should succeed");
    
    let mut interpreter = PhiFlowInterpreter::new();
    let result = interpreter.execute_program(program).await.expect("Execution should succeed");
    
    match result {
        PhiFlowValue::Number(n) => {
            assert!((n - (42.0 + 1.618033988749895 * 2.0)).abs() < 1e-10);
        }
        _ => panic!("Expected numeric result"),
    }
}

#[tokio::test]
async fn test_sacred_frequency_program() {
    let source = r#"
        Sacred(432) {
            let grounding = 432;
            grounding * PHI
        }
    "#;
    
    let mut lexer = PhiFlowLexer::new(source.to_string());
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    let mut parser = PhiFlowParser::new(tokens);
    let program = parser.parse().expect("Parsing should succeed");
    
    let mut interpreter = PhiFlowInterpreter::new();
    let result = interpreter.execute_program(program).await.expect("Execution should succeed");
    
    match result {
        PhiFlowValue::Number(n) => {
            let expected = 432.0 * 1.618033988749895;
            assert!((n - expected).abs() < 1e-10);
        }
        _ => panic!("Expected numeric result"),
    }
}

#[tokio::test]
async fn test_quantum_circuit_creation() {
    use phiflow::quantum::{QuantumCircuit, QuantumGate};
    
    let circuit = QuantumCircuit {
        qubits: 2,
        gates: vec![
            QuantumGate::H(0),
            QuantumGate::CNOT(0, 1),
        ],
        measurements: vec![0, 1],
        metadata: HashMap::new(),
    };
    
    assert_eq!(circuit.qubits, 2);
    assert_eq!(circuit.gates.len(), 2);
    assert_eq!(circuit.measurements.len(), 2);
}

#[tokio::test]
async fn test_quantum_simulator_backend() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let circuit = QuantumCircuit {
        qubits: 2,
        gates: vec![QuantumGate::H(0)],
        measurements: vec![0, 1],
        metadata: HashMap::new(),
    };
    
    let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
        .expect("Circuit execution should succeed");
    
    assert_eq!(result.status, "COMPLETED");
    assert!(result.execution_time >= 0.0);
    assert!(!result.counts.is_empty());
}

#[tokio::test]
async fn test_sacred_frequency_quantum_operation() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let result = backend_manager.execute_sacred_frequency_operation(432, 2, Some("simulator")).await
        .expect("Sacred frequency operation should succeed");
    
    assert_eq!(result.status, "COMPLETED");
    assert!(result.metadata.contains_key("sacred_frequency"));
}

#[tokio::test]
async fn test_phi_harmonic_quantum_operation() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let result = backend_manager.execute_phi_gate(0, 1.618, Some("simulator")).await
        .expect("Phi-harmonic operation should succeed");
    
    assert_eq!(result.status, "COMPLETED");
    assert!(result.metadata.contains_key("phi_power"));
}

#[tokio::test]
async fn test_consciousness_state_creation() {
    let consciousness_state = PhiFlowValue::ConsciousnessState {
        coherence: 0.85,
        clarity: 0.78,
        flow_state: 0.92,
        sacred_frequency: Some(528),
    };
    
    match consciousness_state {
        PhiFlowValue::ConsciousnessState { coherence, clarity, flow_state, sacred_frequency } => {
            assert_eq!(coherence, 0.85);
            assert_eq!(clarity, 0.78);
            assert_eq!(flow_state, 0.92);
            assert_eq!(sacred_frequency, Some(528));
        }
        _ => panic!("Expected consciousness state"),
    }
}

#[tokio::test]
async fn test_quantum_backend_capabilities() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 8,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let capabilities = backend_manager.get_backend_capabilities(Some("simulator")).await
        .expect("Should get capabilities");
    
    assert!(capabilities.max_qubits >= 8);
    assert!(capabilities.supports_sacred_frequencies);
    assert!(capabilities.supports_phi_harmonic);
    assert!(!capabilities.gate_set.is_empty());
}

#[tokio::test]
async fn test_multiple_sacred_frequencies() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let sacred_frequencies = [432, 528, 594, 720, 963];
    
    for frequency in sacred_frequencies {
        let result = backend_manager.execute_sacred_frequency_operation(frequency, 1, Some("simulator")).await
            .expect(&format!("Sacred frequency {} operation should succeed", frequency));
        
        assert_eq!(result.status, "COMPLETED");
        assert!(result.metadata.contains_key("sacred_frequency"));
    }
}

#[tokio::test]
async fn test_phi_powers() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let phi_powers = [1.0, 1.618, 2.618, 4.236]; // φ^0, φ^1, φ^2, φ^3
    
    for phi_power in phi_powers {
        let result = backend_manager.execute_phi_gate(0, phi_power, Some("simulator")).await
            .expect(&format!("Phi-harmonic φ^{} operation should succeed", phi_power));
        
        assert_eq!(result.status, "COMPLETED");
        assert!(result.metadata.contains_key("phi_power"));
    }
}

#[tokio::test]
async fn test_backend_status() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let status = backend_manager.get_backend_status(Some("simulator")).await
        .expect("Should get backend status");
    
    assert!(status.operational);
    assert_eq!(status.pending_jobs, 0);
    assert_eq!(status.queue_length, 0);
}

#[tokio::test]
async fn test_complex_quantum_circuit() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    // Create a more complex circuit with sacred frequencies and phi-harmonic gates
    let circuit = QuantumCircuit {
        qubits: 3,
        gates: vec![
            QuantumGate::H(0),
            QuantumGate::SacredFrequency(1, 528), // Love frequency
            QuantumGate::PhiHarmonic(2, 1.618),   // Golden ratio
            QuantumGate::CNOT(0, 1),
            QuantumGate::CNOT(1, 2),
        ],
        measurements: vec![0, 1, 2],
        metadata: [
            ("experiment_type".to_string(), serde_json::json!("consciousness_quantum_entanglement")),
            ("sacred_frequency".to_string(), serde_json::json!(528)),
            ("phi_power".to_string(), serde_json::json!(1.618)),
        ].iter().cloned().collect(),
    };
    
    let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
        .expect("Complex circuit execution should succeed");
    
    assert_eq!(result.status, "COMPLETED");
    assert!(!result.counts.is_empty());
    assert!(result.metadata.contains_key("experiment_type"));
}

#[tokio::test]
async fn test_error_handling_invalid_backend() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    let circuit = QuantumCircuit {
        qubits: 2,
        gates: vec![QuantumGate::H(0)],
        measurements: vec![0, 1],
        metadata: HashMap::new(),
    };
    
    let result = backend_manager.execute_circuit(circuit, Some("nonexistent_backend")).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_sacred_frequency_validation() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend initialization should succeed");
    
    // Test invalid sacred frequency
    let result = backend_manager.execute_sacred_frequency_operation(999, 1, Some("simulator")).await;
    assert!(result.is_err());
    
    // Test valid sacred frequency
    let result = backend_manager.execute_sacred_frequency_operation(432, 1, Some("simulator")).await;
    assert!(result.is_ok());
}