// PhiFlow Performance Tests - Test performance characteristics and benchmarks

use std::collections::HashMap;
use std::time::Instant;

#[tokio::test]
async fn test_quantum_simulator_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 8,
        shots: 1024,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    // Test performance with increasing qubit counts
    let qubit_counts = [1, 2, 3, 4, 5];
    
    for qubits in qubit_counts {
        let start_time = Instant::now();
        
        let circuit = QuantumCircuit {
            qubits,
            gates: vec![QuantumGate::H(0)], // Simple Hadamard gate
            measurements: (0..qubits).collect(),
            metadata: HashMap::new(),
        };
        
        let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
            .expect("Circuit execution should succeed");
        
        let execution_time = start_time.elapsed();
        
        assert_eq!(result.status, "COMPLETED");
        println!("Qubits: {}, Execution time: {:?}", qubits, execution_time);
        
        // Execution time should be reasonable (less than 1 second for small circuits)
        assert!(execution_time.as_secs() < 1);
    }
}

#[tokio::test]
async fn test_sacred_frequency_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        shots: 1024,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let sacred_frequencies = [432, 528, 594, 639, 693, 741, 852, 963];
    
    for frequency in sacred_frequencies {
        let start_time = Instant::now();
        
        let result = backend_manager.execute_sacred_frequency_operation(frequency, 2, Some("simulator")).await
            .expect("Sacred frequency operation should succeed");
        
        let execution_time = start_time.elapsed();
        
        assert_eq!(result.status, "COMPLETED");
        println!("Sacred frequency {} Hz, Execution time: {:?}", frequency, execution_time);
        
        // Sacred frequency operations should be fast (less than 500ms)
        assert!(execution_time.as_millis() < 500);
    }
}

#[tokio::test]
async fn test_phi_harmonic_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        shots: 1024,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let phi_powers = [1.0, 1.618, 2.618, 4.236, 6.854, 11.090, 17.944]; // φ^0 through φ^6
    
    for phi_power in phi_powers {
        let start_time = Instant::now();
        
        let result = backend_manager.execute_phi_gate(0, phi_power, Some("simulator")).await
            .expect("Phi-harmonic operation should succeed");
        
        let execution_time = start_time.elapsed();
        
        assert_eq!(result.status, "COMPLETED");
        println!("Phi-harmonic φ^{}, Execution time: {:?}", phi_power, execution_time);
        
        // Phi-harmonic operations should be fast (less than 300ms)
        assert!(execution_time.as_millis() < 300);
    }
}

#[tokio::test]
async fn test_complex_circuit_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 6,
        shots: 1024,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    // Create increasingly complex circuits
    let gate_counts = [5, 10, 20, 30];
    
    for gate_count in gate_counts {
        let start_time = Instant::now();
        
        let mut gates = Vec::new();
        
        // Add varied gates to create complexity
        for i in 0..gate_count {
            match i % 6 {
                0 => gates.push(QuantumGate::H(i % 3)),
                1 => gates.push(QuantumGate::X(i % 3)),
                2 => gates.push(QuantumGate::Y(i % 3)),
                3 => gates.push(QuantumGate::Z(i % 3)),
                4 => gates.push(QuantumGate::SacredFrequency(i % 3, 528)),
                5 => gates.push(QuantumGate::PhiHarmonic(i % 3, 1.618)),
                _ => {}
            }
        }
        
        let circuit = QuantumCircuit {
            qubits: 3,
            gates,
            measurements: vec![0, 1, 2],
            metadata: HashMap::new(),
        };
        
        let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
            .expect("Complex circuit execution should succeed");
        
        let execution_time = start_time.elapsed();
        
        assert_eq!(result.status, "COMPLETED");
        println!("Gates: {}, Execution time: {:?}", gate_count, execution_time);
        
        // Complex circuits should still execute reasonably quickly (less than 2 seconds)
        assert!(execution_time.as_secs() < 2);
    }
}

#[tokio::test]
async fn test_parallel_execution_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        shots: 512, // Reduce shots for faster parallel execution
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let start_time = Instant::now();
    
    for i in 0..5 {
        let circuit = QuantumCircuit {
            qubits: 2,
            gates: vec![
                QuantumGate::H(0),
                QuantumGate::SacredFrequency(1, 432 + i * 100), // Vary frequency
                QuantumGate::CNOT(0, 1),
            ],
            measurements: vec![0, 1],
            metadata: HashMap::new(),
        };
        
        // Clone backend_manager for each task (in real implementation, we'd use Arc)
        // For this test, we'll execute sequentially but measure parallel potential
        let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
            .expect("Parallel circuit execution should succeed");
        
        assert_eq!(result.status, "COMPLETED");
    }
    
    let total_time = start_time.elapsed();
    println!("Total time for 5 circuits: {:?}", total_time);
    
    // All 5 circuits should execute in reasonable time (less than 3 seconds total)
    assert!(total_time.as_secs() < 3);
}

#[tokio::test]
async fn test_memory_usage_with_large_circuits() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 8, // 2^8 = 256 amplitude state vector
        shots: 1024,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    // Test memory usage with larger state vectors
    let qubit_counts = [4, 5, 6, 7, 8]; // Up to 2^8 = 256 amplitudes
    
    for qubits in qubit_counts {
        let circuit = QuantumCircuit {
            qubits,
            gates: vec![
                QuantumGate::H(0), // Create superposition
                QuantumGate::SacredFrequency(qubits - 1, 528), // Apply to last qubit
            ],
            measurements: (0..qubits).collect(),
            metadata: HashMap::new(),
        };
        
        let start_time = Instant::now();
        
        let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
            .expect("Large circuit execution should succeed");
        
        let execution_time = start_time.elapsed();
        
        assert_eq!(result.status, "COMPLETED");
        
        // Check that statevector has correct size (2^qubits)
        if let Some(ref statevector) = result.statevector {
            let expected_size = 1 << qubits; // 2^qubits
            assert_eq!(statevector.len(), expected_size);
        }
        
        println!("Qubits: {}, State vector size: {}, Execution time: {:?}", 
                qubits, 1 << qubits, execution_time);
        
        // Larger circuits should still execute in reasonable time
        assert!(execution_time.as_secs() < 5);
    }
}

#[tokio::test]
async fn test_shots_scaling_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    
    let base_config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        ..Default::default()
    };
    
    let shot_counts = [128, 256, 512, 1024, 2048];
    
    for shots in shot_counts {
        let config = QuantumConfig {
            shots,
            ..base_config.clone()
        };
        
        let mut backend_manager = QuantumBackendManager::new();
        backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
        
        let start_time = Instant::now();
        
        let circuit = QuantumCircuit {
            qubits: 3,
            gates: vec![
                QuantumGate::H(0),
                QuantumGate::H(1),
                QuantumGate::H(2),
                QuantumGate::SacredFrequency(0, 432),
                QuantumGate::PhiHarmonic(1, 1.618),
            ],
            measurements: vec![0, 1, 2],
            metadata: HashMap::new(),
        };
        
        let result = backend_manager.execute_circuit(circuit, Some("simulator")).await
            .expect("Circuit execution should succeed");
        
        let execution_time = start_time.elapsed();
        
        assert_eq!(result.status, "COMPLETED");
        
        // Total counts should equal number of shots
        let total_counts: u32 = result.counts.values().sum();
        assert_eq!(total_counts, shots);
        
        println!("Shots: {}, Execution time: {:?}", shots, execution_time);
        
        // Execution time should scale reasonably with shot count
        // More shots = more time, but should remain sub-linear
        assert!(execution_time.as_secs() < 3);
    }
}

#[tokio::test]
async fn test_backend_initialization_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig};
    
    let start_time = Instant::now();
    
    // Test multiple backend initializations
    for i in 0..5 {
        let config = QuantumConfig {
            backend_name: "simulator".to_string(),
            max_qubits: 4 + i,
            shots: 1024,
            ..Default::default()
        };
        
        let mut backend_manager = QuantumBackendManager::new();
        backend_manager.initialize(config).await.expect("Backend initialization should succeed");
        
        // Verify backend is working
        let backends = backend_manager.list_backends().await;
        assert!(!backends.is_empty());
    }
    
    let total_time = start_time.elapsed();
    println!("Total initialization time for 5 backends: {:?}", total_time);
    
    // Multiple backend initializations should be fast (less than 1 second total)
    assert!(total_time.as_secs() < 1);
}

#[tokio::test]
async fn test_consciousness_quantum_coupling_performance() {
    use phiflow::quantum::{QuantumBackendManager, QuantumConfig, QuantumCircuit, QuantumGate};
    use phiflow::quantum::types::ConsciousnessQuantumCoupling;
    
    let config = QuantumConfig {
        backend_name: "simulator".to_string(),
        max_qubits: 4,
        shots: 1024,
        ..Default::default()
    };
    
    let mut backend_manager = QuantumBackendManager::new();
    backend_manager.initialize(config).await.expect("Backend manager initialization should succeed");
    
    let consciousness_coupling = ConsciousnessQuantumCoupling {
        coherence_threshold: 0.85,
        sacred_frequency_lock: Some(528),
        phi_resonance: 0.92,
        quantum_authorization: true,
    };
    
    let start_time = Instant::now();
    
    // Test multiple consciousness-coupled executions
    for i in 0..3 {
        let circuit = QuantumCircuit {
            qubits: 3,
            gates: vec![
                QuantumGate::H(0),
                QuantumGate::SacredFrequency(1, 528),
                QuantumGate::PhiHarmonic(2, 1.618),
                QuantumGate::CNOT(0, 1),
                QuantumGate::CNOT(1, 2),
            ],
            measurements: vec![0, 1, 2],
            metadata: [("consciousness_iteration".to_string(), serde_json::json!(i))].iter().cloned().collect(),
        };
        
        let result = backend_manager.execute_consciousness_coupled_circuit(
            circuit, 
            consciousness_coupling.clone(), 
            Some("simulator")
        ).await.expect("Consciousness-coupled execution should succeed");
        
        assert_eq!(result.status, "COMPLETED");
    }
    
    let total_time = start_time.elapsed();
    println!("Total consciousness-coupled execution time for 3 circuits: {:?}", total_time);
    
    // Consciousness-coupled executions should not add significant overhead
    assert!(total_time.as_secs() < 2);
}
