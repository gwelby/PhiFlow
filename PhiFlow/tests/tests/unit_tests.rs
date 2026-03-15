// PhiFlow Unit Tests - Test individual components in isolation

use phiflow::*;
use std::collections::HashMap;

#[test]
fn test_lexer_basic_tokens() {
    let source = "let x = 42 + PHI;";
    let mut lexer = PhiFlowLexer::new(source.to_string());
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    assert!(tokens.len() > 5); // Should have at least: let, x, =, 42, +, PHI, ;
}

#[test]
fn test_lexer_sacred_frequency() {
    let source = "Sacred(432)";
    let mut lexer = PhiFlowLexer::new(source.to_string());
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    assert!(tokens.len() >= 4); // Sacred, (, 432, )
}

#[test]
fn test_lexer_consciousness_keywords() {
    let source = "consciousness.coherence > 0.8";
    let mut lexer = PhiFlowLexer::new(source.to_string());
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    assert!(tokens.len() >= 5); // consciousness, ., coherence, >, 0.8
}

#[test]
fn test_phi_constant() {
    let phi = 1.618033988749895_f64;
    assert!((phi - 1.618033988749895).abs() < 1e-15);
    
    // Test phi properties
    let phi_squared = phi * phi;
    let phi_plus_one = phi + 1.0;
    assert!((phi_squared - phi_plus_one).abs() < 1e-10);
}

#[test]
fn test_sacred_frequencies() {
    use phiflow::quantum::types::{SACRED_FREQUENCIES, is_sacred_frequency, get_nearest_sacred_frequency};
    
    // Test known sacred frequencies
    assert!(is_sacred_frequency(432));
    assert!(is_sacred_frequency(528));
    assert!(is_sacred_frequency(963));
    
    // Test invalid frequency
    assert!(!is_sacred_frequency(999));
    
    // Test nearest frequency
    assert_eq!(get_nearest_sacred_frequency(430), 432);
    assert_eq!(get_nearest_sacred_frequency(530), 528);
}

#[test]
fn test_frequency_to_quantum_angle() {
    use phiflow::quantum::types::frequency_to_quantum_angle;
    
    let angle_432 = frequency_to_quantum_angle(432);
    let angle_528 = frequency_to_quantum_angle(528);
    
    assert!(angle_432 > 0.0);
    assert!(angle_528 > angle_432); // Higher frequency should give larger angle
}

#[test]
fn test_phi_power_to_angle() {
    use phiflow::quantum::types::phi_power_to_angle;
    
    let angle_1 = phi_power_to_angle(1.0);
    let angle_phi = phi_power_to_angle(1.618);
    let angle_phi_squared = phi_power_to_angle(2.618);
    
    assert!(angle_1 > 0.0);
    assert!(angle_phi > angle_1);
    assert!(angle_phi_squared > angle_phi);
}

#[test]
fn test_phi_resonance_calculation() {
    use phiflow::quantum::types::calculate_phi_resonance;
    
    let resonance_432 = calculate_phi_resonance(432);
    let resonance_700 = calculate_phi_resonance(700); // Closer to PHI * 432
    
    assert!(resonance_432 > 0.0 && resonance_432 <= 1.0);
    assert!(resonance_700 > 0.0 && resonance_700 <= 1.0);
}

#[test]
fn test_quantum_gate_types() {
    use phiflow::quantum::{QuantumGate};
    
    let gates = vec![
        QuantumGate::H(0),
        QuantumGate::X(1),
        QuantumGate::CNOT(0, 1),
        QuantumGate::SacredFrequency(0, 432),
        QuantumGate::PhiHarmonic(1, 1.618),
    ];
    
    assert_eq!(gates.len(), 5);
    
    match &gates[0] {
        QuantumGate::H(qubit) => assert_eq!(*qubit, 0),
        _ => panic!("Expected Hadamard gate"),
    }
    
    match &gates[3] {
        QuantumGate::SacredFrequency(qubit, freq) => {
            assert_eq!(*qubit, 0);
            assert_eq!(*freq, 432);
        }
        _ => panic!("Expected Sacred Frequency gate"),
    }
}

#[test]
fn test_quantum_circuit_creation() {
    use phiflow::quantum::{QuantumCircuit, QuantumGate};
    
    let circuit = QuantumCircuit {
        qubits: 3,
        gates: vec![
            QuantumGate::H(0),
            QuantumGate::CNOT(0, 1),
            QuantumGate::CNOT(1, 2),
        ],
        measurements: vec![0, 1, 2],
        metadata: HashMap::new(),
    };
    
    assert_eq!(circuit.qubits, 3);
    assert_eq!(circuit.gates.len(), 3);
    assert_eq!(circuit.measurements.len(), 3);
}

#[test]
fn test_quantum_result_structure() {
    use phiflow::quantum::QuantumResult;
    use num_complex::Complex64;
    
    let result = QuantumResult {
        job_id: "test_job_123".to_string(),
        status: "COMPLETED".to_string(),
        counts: [("00".to_string(), 512), ("11".to_string(), 512)].iter().cloned().collect(),
        statevector: Some(vec![
            Complex64::new(0.707, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.707, 0.0),
        ]),
        execution_time: 0.123,
        backend_name: "simulator".to_string(),
        metadata: HashMap::new(),
    };
    
    assert_eq!(result.job_id, "test_job_123");
    assert_eq!(result.status, "COMPLETED");
    assert_eq!(result.counts.len(), 2);
    assert!(result.statevector.is_some());
    assert_eq!(result.statevector.unwrap().len(), 4);
}

#[test]
fn test_quantum_capabilities() {
    use phiflow::quantum::QuantumCapabilities;
    
    let capabilities = QuantumCapabilities {
        max_qubits: 32,
        gate_set: vec!["h".to_string(), "x".to_string(), "cx".to_string()],
        supports_sacred_frequencies: true,
        supports_phi_harmonic: true,
        coupling_map: None,
        basis_gates: vec!["u1".to_string(), "u2".to_string(), "u3".to_string(), "cx".to_string()],
    };
    
    assert_eq!(capabilities.max_qubits, 32);
    assert!(capabilities.supports_sacred_frequencies);
    assert!(capabilities.supports_phi_harmonic);
    assert_eq!(capabilities.gate_set.len(), 3);
    assert_eq!(capabilities.basis_gates.len(), 4);
}

#[test]
fn test_backend_status() {
    use phiflow::quantum::BackendStatus;
    
    let status = BackendStatus {
        operational: true,
        pending_jobs: 5,
        queue_length: 10,
        status_msg: "Online".to_string(),
        last_update: "2024-01-01T00:00:00Z".to_string(),
    };
    
    assert!(status.operational);
    assert_eq!(status.pending_jobs, 5);
    assert_eq!(status.queue_length, 10);
    assert_eq!(status.status_msg, "Online");
}

#[test]
fn test_consciousness_quantum_coupling() {
    use phiflow::quantum::types::ConsciousnessQuantumCoupling;
    
    let coupling = ConsciousnessQuantumCoupling {
        coherence_threshold: 0.85,
        sacred_frequency_lock: Some(528),
        phi_resonance: 0.92,
        quantum_authorization: true,
    };
    
    assert_eq!(coupling.coherence_threshold, 0.85);
    assert_eq!(coupling.sacred_frequency_lock, Some(528));
    assert_eq!(coupling.phi_resonance, 0.92);
    assert!(coupling.quantum_authorization);
}

#[test]
fn test_sacred_frequency_operation() {
    use phiflow::quantum::types::SacredFrequencyOperation;
    
    let operation = SacredFrequencyOperation {
        frequency: 432,
        qubits: vec![0, 1, 2],
        duration: 60.0,
        consciousness_coupling: true,
    };
    
    assert_eq!(operation.frequency, 432);
    assert_eq!(operation.qubits.len(), 3);
    assert_eq!(operation.duration, 60.0);
    assert!(operation.consciousness_coupling);
}

#[test]
fn test_phi_harmonic_operation() {
    use phiflow::quantum::types::{PhiHarmonicOperation, PhiAxis};
    
    let operation = PhiHarmonicOperation {
        qubit: 0,
        phi_power: 1.618,
        axis: PhiAxis::Z,
        coupling_strength: 0.95,
    };
    
    assert_eq!(operation.qubit, 0);
    assert_eq!(operation.phi_power, 1.618);
    assert_eq!(operation.coupling_strength, 0.95);
    
    match operation.axis {
        PhiAxis::Z => {}, // Expected
        _ => panic!("Expected Z axis"),
    }
}

#[test]
fn test_phiflow_value_types() {
    let values = vec![
        PhiFlowValue::Number(42.0),
        PhiFlowValue::String("test".to_string()),
        PhiFlowValue::Boolean(true),
        PhiFlowValue::SacredFrequency(432),
        PhiFlowValue::ConsciousnessState {
            coherence: 0.85,
            clarity: 0.78,
            flow_state: 0.92,
            sacred_frequency: Some(528),
        },
        PhiFlowValue::Nil,
    ];
    
    assert_eq!(values.len(), 6);
    
    match &values[0] {
        PhiFlowValue::Number(n) => assert_eq!(*n, 42.0),
        _ => panic!("Expected number"),
    }
    
    match &values[3] {
        PhiFlowValue::SacredFrequency(f) => assert_eq!(*f, 432),
        _ => panic!("Expected sacred frequency"),
    }
    
    match &values[4] {
        PhiFlowValue::ConsciousnessState { coherence, clarity, flow_state, sacred_frequency } => {
            assert_eq!(*coherence, 0.85);
            assert_eq!(*clarity, 0.78);
            assert_eq!(*flow_state, 0.92);
            assert_eq!(*sacred_frequency, Some(528));
        }
        _ => panic!("Expected consciousness state"),
    }
}

#[test]
fn test_quantum_config_defaults() {
    use phiflow::quantum::QuantumConfig;
    
    let config = QuantumConfig::default();
    
    assert_eq!(config.backend_name, "simulator");
    assert_eq!(config.max_qubits, 32);
    assert_eq!(config.shots, 1024);
    assert_eq!(config.timeout_seconds, 300);
    assert!(config.api_token.is_none());
}

#[test]
fn test_quantum_config_custom() {
    use phiflow::quantum::QuantumConfig;
    
    let config = QuantumConfig {
        backend_name: "ibm_quantum".to_string(),
        api_token: Some("test_token".to_string()),
        hub: Some("test_hub".to_string()),
        group: Some("test_group".to_string()),
        project: Some("test_project".to_string()),
        max_qubits: 16,
        shots: 2048,
        timeout_seconds: 600,
    };
    
    assert_eq!(config.backend_name, "ibm_quantum");
    assert_eq!(config.api_token, Some("test_token".to_string()));
    assert_eq!(config.hub, Some("test_hub".to_string()));
    assert_eq!(config.max_qubits, 16);
    assert_eq!(config.shots, 2048);
    assert_eq!(config.timeout_seconds, 600);
}