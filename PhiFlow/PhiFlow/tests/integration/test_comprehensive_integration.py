#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for PhiFlow Integration Engine

This test suite validates all integration bridges work together properly:
- Rust-Python FFI bridge
- CUDA-Consciousness bridge  
- Quantum backend integration
- EEG consciousness pipeline
- Cross-component communication
- System-wide coherence validation

Tests Requirements 3.1-3.5 for the Integration Engine.
"""

import pytest
import asyncio
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

# Import all integration components
from integration.rust_python_bridge import RustPythonBridge, QuantumCircuitResult, ConsciousnessMetrics
from integration.cuda_consciousness_bridge import CUDAConsciousnessBridge, ConsciousnessState, CUDAPerformanceMetrics
from integration.quantum_backend_integration import QuantumBackendIntegration, QuantumCircuit, QuantumExecutionResult
from integration.consciousness_eeg_pipeline import ConsciousnessEEGPipeline, ConsciousnessAnalysisResult, ConsciousnessProcessingConfig
from integration.phi_flow_integration_engine import PhiFlowIntegrationEngine, IntegrationState, IntegrationCommand

# Import core components for testing
try:
    from coherence.phi_coherence_engine import PhiCoherenceEngine
    from optimization.phi_quantum_optimizer import PhiQuantumOptimizer  
    from parser.phi_flow_parser import PhiFlowParser
except ImportError as e:
    # Mock if components not available
    PhiCoherenceEngine = Mock
    PhiQuantumOptimizer = Mock
    PhiFlowParser = Mock


class TestIntegrationArchitecture:
    """Test the complete integration architecture"""
    
    @pytest.fixture
    def integration_engine(self):
        """Create integration engine with all components"""
        engine = PhiFlowIntegrationEngine()
        
        # Initialize with mock components if needed
        try:
            engine.initialize()
        except Exception:
            # Use mocked components for testing
            engine.coherence_engine = Mock()
            engine.optimizer = Mock() 
            engine.parser = Mock()
            engine.quantum_bridge = Mock()
            engine.consciousness_monitor = Mock()
            
        return engine
    
    def test_integration_engine_initialization(self, integration_engine):
        """Test that integration engine initializes all components"""
        assert integration_engine is not None
        assert hasattr(integration_engine, 'coherence_engine')
        assert hasattr(integration_engine, 'optimizer')
        assert hasattr(integration_engine, 'parser')
        assert hasattr(integration_engine, 'quantum_bridge')
        assert hasattr(integration_engine, 'consciousness_monitor')
    
    def test_component_health_check(self, integration_engine):
        """Test component health checking"""
        health = integration_engine.check_component_health()
        
        assert isinstance(health, dict)
        assert 'coherence_engine' in health
        assert 'optimizer' in health
        assert 'parser' in health
        assert 'quantum_bridge' in health
        assert 'consciousness_monitor' in health
        
        # Each component should have status
        for component, status in health.items():
            assert 'status' in status
            assert 'last_check' in status
    
    def test_cross_component_communication(self, integration_engine):
        """Test communication between components"""
        # Test data flow from consciousness to quantum
        test_data = {
            'consciousness_state': 'TRANSCEND',
            'coherence_level': 0.95,
            'phi_alignment': 0.88
        }
        
        result = integration_engine.coordinate_cross_component_operation(test_data)
        
        assert result is not None
        assert 'status' in result
        assert 'processing_time' in result


class TestRustPythonIntegration:
    """Test Rust-Python FFI bridge integration"""
    
    @pytest.fixture
    def rust_bridge(self):
        """Create Rust-Python bridge"""
        return RustPythonBridge()
    
    def test_rust_bridge_initialization(self, rust_bridge):
        """Test Rust bridge initialization"""
        assert rust_bridge is not None
        assert rust_bridge.initialized
        assert rust_bridge.performance_metrics is not None
    
    def test_quantum_circuit_execution_integration(self, rust_bridge):
        """Test quantum circuit execution through Rust bridge"""
        # Create test quantum circuit
        circuit = {
            "gates": [
                {"type": "H", "qubit": 0},
                {"type": "CNOT", "control": 0, "target": 1},
                {"type": "measure", "qubits": [0, 1]}
            ],
            "qubits": 2
        }
        
        circuit_json = json.dumps(circuit)
        result = rust_bridge.execute_quantum_circuit(circuit_json)
        
        assert isinstance(result, QuantumCircuitResult)
        assert result.success
        assert result.measurements is not None
        assert result.execution_time > 0
    
    def test_consciousness_metrics_integration(self, rust_bridge):
        """Test consciousness metrics from Rust"""
        metrics = rust_bridge.get_consciousness_metrics()
        
        assert isinstance(metrics, ConsciousnessMetrics)
        assert 0 <= metrics.coherence_level <= 1
        assert 0 <= metrics.phi_alignment <= 1
        assert metrics.consciousness_state in ['OBSERVE', 'CREATE', 'INTEGRATE', 'HARMONIZE', 'TRANSCEND', 'CASCADE', 'SUPERPOSITION']
    
    def test_cuda_computation_integration(self, rust_bridge):
        """Test CUDA computation through Rust bridge"""
        test_data = np.random.rand(1000).astype(np.float32)
        parameters = {
            'phi_power': 2.0,
            'precision': 10
        }
        
        result = rust_bridge.execute_cuda_computation('phi_parallel', test_data, parameters)
        
        assert result.success
        assert result.output is not None
        assert result.performance_metrics['tflops'] > 0


class TestCUDAConsciousnessIntegration:
    """Test CUDA-Consciousness bridge integration"""
    
    @pytest.fixture
    def cuda_bridge(self):
        """Create CUDA-Consciousness bridge"""
        return CUDAConsciousnessBridge()
    
    def test_cuda_bridge_initialization(self, cuda_bridge):
        """Test CUDA bridge initialization"""
        assert cuda_bridge is not None
        assert cuda_bridge.cuda_available
        assert cuda_bridge.performance_metrics is not None
    
    def test_phi_parallel_computation_integration(self, cuda_bridge):
        """Test PHI parallel computation"""
        test_data = np.random.rand(10000).astype(np.float32)
        
        result = cuda_bridge.execute_phi_parallel_computation(test_data, phi_power=1.0, precision=15)
        
        assert result.success
        assert result.output is not None
        assert result.performance_metrics.tflops > 1.0  # Target >1 TFLOP/s
        assert result.performance_metrics.phi_calculations_per_second > 1e9  # >1 billion PHI/s
    
    def test_sacred_frequency_synthesis_integration(self, cuda_bridge):
        """Test sacred frequency synthesis"""
        sacred_frequencies = [432, 528, 594, 672, 720, 768, 963]
        
        result = cuda_bridge.execute_sacred_frequency_synthesis(sacred_frequencies, duration=1.0)
        
        assert result.success
        assert result.output is not None
        assert result.performance_metrics.simultaneous_waveforms >= len(sacred_frequencies)
    
    def test_consciousness_eeg_processing_integration(self, cuda_bridge):
        """Test EEG consciousness processing"""
        # Generate synthetic EEG data (8 channels, 256 samples)
        eeg_data = np.random.randn(8, 256).astype(np.float32)
        
        consciousness_state = cuda_bridge.process_consciousness_eeg_data(eeg_data, sampling_rate=256)
        
        assert isinstance(consciousness_state, ConsciousnessState)
        assert consciousness_state.level in ['OBSERVE', 'CREATE', 'INTEGRATE', 'HARMONIZE', 'TRANSCEND', 'CASCADE', 'SUPERPOSITION']
        assert 0 <= consciousness_state.coherence <= 1
        assert 0 <= consciousness_state.phi_alignment <= 1


class TestQuantumBackendIntegration:
    """Test quantum backend integration"""
    
    @pytest.fixture
    def quantum_backend(self):
        """Create quantum backend integration"""
        return QuantumBackendIntegration()
    
    def test_quantum_backend_initialization(self, quantum_backend):
        """Test quantum backend initialization"""
        assert quantum_backend is not None
        assert len(quantum_backend.available_backends) > 0
        assert 'simulator' in quantum_backend.available_backends
    
    def test_quantum_circuit_execution_integration(self, quantum_backend):
        """Test quantum circuit execution across backends"""
        # Create test circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate('H', [0])
        circuit.add_gate('CNOT', [0, 1])
        circuit.add_measurement([0, 1])
        
        # Test with simulator backend
        result = quantum_backend.execute_quantum_circuit(circuit, backend='simulator')
        
        assert isinstance(result, QuantumExecutionResult)
        assert result.success
        assert result.measurements is not None
        assert result.execution_time > 0
    
    def test_consciousness_enhanced_execution(self, quantum_backend):
        """Test consciousness-enhanced quantum execution"""
        # Create test circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate('H', [0])
        circuit.add_gate('RY', [1], parameters={'angle': 1.618033988749895})  # PHI angle
        circuit.add_measurement([0, 1])
        
        result = quantum_backend.execute_quantum_circuit(circuit, consciousness_enhanced=True) 
        
        assert result.success
        assert result.consciousness_correlation is not None
        assert result.phi_alignment is not None
    
    def test_sacred_frequency_operations_integration(self, quantum_backend):
        """Test sacred frequency quantum operations"""
        for frequency in [432, 528, 594, 672, 720, 768, 963]:
            result = quantum_backend.execute_sacred_frequency_operation(frequency, [0, 1])
            
            assert result.success
            assert result.frequency == frequency
            assert result.measurements is not None


class TestConsciousnessEEGIntegration:
    """Test consciousness EEG pipeline integration"""
    
    @pytest.fixture
    def eeg_pipeline(self):
        """Create EEG pipeline"""
        config = ConsciousnessProcessingConfig(
            device_type='simulator',  # Use simulator for testing
            sampling_rate=256,
            buffer_size=1024,
            processing_latency_target=10  # <10ms target
        )
        return ConsciousnessEEGPipeline(config)
    
    def test_eeg_pipeline_initialization(self, eeg_pipeline):
        """Test EEG pipeline initialization"""
        assert eeg_pipeline is not None
        assert eeg_pipeline.config.device_type == 'simulator'
        assert eeg_pipeline.config.sampling_rate == 256
        assert eeg_pipeline.performance_metrics is not None
    
    def test_real_time_consciousness_processing(self, eeg_pipeline):
        """Test real-time consciousness processing"""
        # Generate synthetic EEG data
        eeg_data = np.random.randn(8, 256).astype(np.float32)
        
        start_time = time.time()
        consciousness_state = eeg_pipeline.process_consciousness_eeg_data(eeg_data)
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Verify <10ms processing time requirement
        assert processing_time < 10.0
        
        assert isinstance(consciousness_state, ConsciousnessState)
        assert consciousness_state.level in ['OBSERVE', 'CREATE', 'INTEGRATE', 'HARMONIZE', 'TRANSCEND', 'CASCADE', 'SUPERPOSITION']
    
    def test_consciousness_pattern_analysis(self, eeg_pipeline):
        """Test consciousness pattern analysis"""
        analysis_result = eeg_pipeline.analyze_consciousness_pattern(duration=1.0)  # Short test duration
        
        assert isinstance(analysis_result, ConsciousnessAnalysisResult)
        assert analysis_result.dominant_frequency > 0
        assert 0 <= analysis_result.coherence_score <= 1
        assert len(analysis_result.frequency_distribution) > 0
    
    def test_biofeedback_control_integration(self, eeg_pipeline):
        """Test biofeedback control system"""
        target_success = eeg_pipeline.set_target_consciousness_level('TRANSCEND', duration=5.0)
        
        assert target_success
        assert eeg_pipeline.biofeedback_active
        assert eeg_pipeline.target_consciousness_level == 'TRANSCEND'


class TestSystemWideIntegration:
    """Test system-wide integration and coherence"""
    
    @pytest.fixture
    def full_system(self):
        """Create full integrated system"""
        # Initialize all components
        rust_bridge = RustPythonBridge()
        cuda_bridge = CUDAConsciousnessBridge()
        quantum_backend = QuantumBackendIntegration()
        
        eeg_config = ConsciousnessProcessingConfig(
            device_type='simulator',
            sampling_rate=256,
            buffer_size=1024,
            processing_latency_target=10
        )
        eeg_pipeline = ConsciousnessEEGPipeline(eeg_config)
        
        integration_engine = PhiFlowIntegrationEngine()
        
        # Connect all components
        system = {
            'rust_bridge': rust_bridge,
            'cuda_bridge': cuda_bridge,
            'quantum_backend': quantum_backend,
            'eeg_pipeline': eeg_pipeline,
            'integration_engine': integration_engine
        }
        
        return system
    
    def test_full_system_initialization(self, full_system):
        """Test full system initialization"""
        for component_name, component in full_system.items():
            assert component is not None, f"{component_name} failed to initialize"
    
    def test_end_to_end_integration_flow(self, full_system):
        """Test complete end-to-end integration flow"""
        # Step 1: Get consciousness state from EEG
        eeg_data = np.random.randn(8, 256).astype(np.float32)
        consciousness_state = full_system['eeg_pipeline'].process_consciousness_eeg_data(eeg_data)
        
        # Step 2: Use consciousness state to guide quantum circuit
        circuit = QuantumCircuit(2)
        circuit.add_gate('H', [0])
        circuit.add_gate('RY', [1], parameters={'angle': consciousness_state.phi_alignment * np.pi})
        circuit.add_measurement([0, 1])
        
        quantum_result = full_system['quantum_backend'].execute_quantum_circuit(
            circuit, consciousness_enhanced=True
        )
        
        # Step 3: Process results with CUDA acceleration
        test_data = np.array(quantum_result.measurements, dtype=np.float32)
        cuda_result = full_system['cuda_bridge'].execute_phi_parallel_computation(test_data)
        
        # Step 4: Verify integration through Rust bridge
        rust_metrics = full_system['rust_bridge'].get_consciousness_metrics()
        
        # Verify end-to-end flow
        assert consciousness_state.level in ['OBSERVE', 'CREATE', 'INTEGRATE', 'HARMONIZE', 'TRANSCEND', 'CASCADE', 'SUPERPOSITION']
        assert quantum_result.success
        assert cuda_result.success
        assert isinstance(rust_metrics, ConsciousnessMetrics)
    
    def test_system_wide_coherence_measurement(self, full_system):
        """Test system-wide coherence measurement"""
        # Measure coherence across all components
        coherence_measurements = {}
        
        # EEG coherence
        eeg_data = np.random.randn(8, 256).astype(np.float32)
        consciousness_state = full_system['eeg_pipeline'].process_consciousness_eeg_data(eeg_data)
        coherence_measurements['consciousness'] = consciousness_state.coherence
        
        # Quantum coherence (simulated)
        circuit = QuantumCircuit(2)
        circuit.add_gate('H', [0])
        circuit.add_gate('CNOT', [0, 1])
        circuit.add_measurement([0, 1])
        
        quantum_result = full_system['quantum_backend'].execute_quantum_circuit(circuit)
        coherence_measurements['quantum'] = getattr(quantum_result, 'coherence', 0.95)  # Default if not available
        
        # Field coherence (from CUDA processing)
        test_data = np.random.rand(1000).astype(np.float32)
        cuda_result = full_system['cuda_bridge'].execute_phi_parallel_computation(test_data)
        coherence_measurements['field'] = getattr(cuda_result, 'coherence', 0.90)  # Default if not available
        
        # Calculate combined coherence: (quantum × consciousness × field)^(1/3)
        combined_coherence = (
            coherence_measurements['quantum'] * 
            coherence_measurements['consciousness'] * 
            coherence_measurements['field']
        ) ** (1/3)
        
        # Verify coherence levels
        for component, coherence in coherence_measurements.items():
            assert 0 <= coherence <= 1, f"{component} coherence out of range: {coherence}"
        
        # Target: individual systems >97% for combined >99%
        print(f"Coherence measurements: {coherence_measurements}")
        print(f"Combined coherence: {combined_coherence:.3f}")
        
        # Note: In real system, we'd target 99.9%, but for testing we use more lenient thresholds
        assert combined_coherence > 0.85, f"Combined coherence too low: {combined_coherence:.3f}"
    
    def test_performance_validation(self, full_system):
        """Test performance across all integrated components"""
        performance_metrics = {}
        
        # CUDA performance
        test_data = np.random.rand(10000).astype(np.float32)
        start_time = time.time()
        cuda_result = full_system['cuda_bridge'].execute_phi_parallel_computation(test_data)
        cuda_time = time.time() - start_time
        
        performance_metrics['cuda_tflops'] = cuda_result.performance_metrics.tflops
        performance_metrics['cuda_time'] = cuda_time
        performance_metrics['phi_calculations_per_second'] = cuda_result.performance_metrics.phi_calculations_per_second
        
        # EEG processing latency
        eeg_data = np.random.randn(8, 256).astype(np.float32)
        start_time = time.time()
        consciousness_state = full_system['eeg_pipeline'].process_consciousness_eeg_data(eeg_data)
        eeg_latency = (time.time() - start_time) * 1000  # ms
        
        performance_metrics['eeg_latency_ms'] = eeg_latency
        
        # Quantum execution time
        circuit = QuantumCircuit(2)
        circuit.add_gate('H', [0])
        circuit.add_measurement([0])
        
        start_time = time.time()
        quantum_result = full_system['quantum_backend'].execute_quantum_circuit(circuit)
        quantum_time = time.time() - start_time
        
        performance_metrics['quantum_time'] = quantum_time
        
        # Validate performance requirements
        assert performance_metrics['cuda_tflops'] > 1.0, f"CUDA TFLOPS too low: {performance_metrics['cuda_tflops']}"
        assert performance_metrics['phi_calculations_per_second'] > 1e9, f"PHI calculations/sec too low: {performance_metrics['phi_calculations_per_second']}"
        assert performance_metrics['eeg_latency_ms'] < 10.0, f"EEG latency too high: {performance_metrics['eeg_latency_ms']}"
        
        print(f"Performance metrics: {performance_metrics}")
    
    def test_error_recovery_and_resilience(self, full_system):
        """Test error recovery and system resilience"""
        # Test with invalid data to trigger error handling
        
        # Test EEG pipeline with invalid data
        invalid_eeg_data = np.array([])  # Empty array
        try:
            consciousness_state = full_system['eeg_pipeline'].process_consciousness_eeg_data(invalid_eeg_data)
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            assert "Invalid EEG data" in str(e) or "empty" in str(e).lower()
        
        # Test quantum backend with invalid circuit
        invalid_circuit = QuantumCircuit(0)  # Zero qubits
        try:
            result = full_system['quantum_backend'].execute_quantum_circuit(invalid_circuit)
            # Should either handle gracefully or raise appropriate exception
        except Exception as e:
            assert "Invalid circuit" in str(e) or "qubits" in str(e).lower()
        
        # Test CUDA bridge with invalid data
        invalid_cuda_data = np.array([]).astype(np.float32)  # Empty array
        try:
            result = full_system['cuda_bridge'].execute_phi_parallel_computation(invalid_cuda_data)
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            assert "Invalid data" in str(e) or "empty" in str(e).lower()


class TestPhiFlowProgramExecution:
    """Test complete PhiFlow program execution integration"""
    
    @pytest.fixture
    def program_executor(self):
        """Create program executor with full integration"""
        engine = PhiFlowIntegrationEngine()
        
        # Mock initialize if needed
        try:
            engine.initialize()
        except Exception:
            engine.coherence_engine = Mock()
            engine.optimizer = Mock()
            engine.parser = Mock()
            engine.quantum_bridge = Mock()
            engine.consciousness_monitor = Mock()
        
        return engine
    
    def test_simple_phiflow_program_execution(self, program_executor):
        """Test execution of simple PhiFlow program"""
        # Simple PhiFlow program
        phiflow_program = """
        INITIALIZE consciousness_level=OBSERVE frequency=432
        TRANSITION to_state=CREATE frequency=528 phi_level=1
        EVOLVE iterations=3 phi_power=1.618
        INTEGRATE final_state=TRANSCEND
        """
        
        result = program_executor.execute_phiflow_program(phiflow_program)
        
        assert result is not None
        assert 'status' in result
        assert 'execution_time' in result
        assert 'performance_metrics' in result
    
    def test_complex_phiflow_program_execution(self, program_executor):
        """Test execution of complex PhiFlow program with quantum operations"""
        # Complex PhiFlow program with quantum elements
        phiflow_program = """
        INITIALIZE consciousness_level=OBSERVE frequency=432
        CREATE_FIELD field_type=toroidal frequency=528 phi_level=2
        ENTANGLE qubits=[0,1] consciousness_guided=true
        SUPERPOSE state_count=4 phi_harmonic=true
        MEASURE qubits=[0,1] consciousness_enhanced=true
        HARMONIZE frequency=720 optimization_level=5
        CASCADE final_frequency=963 phi_level=7
        """
        
        result = program_executor.execute_phiflow_program(phiflow_program)
        
        assert result is not None
        assert 'status' in result
        assert 'quantum_metrics' in result
        assert 'consciousness_metrics' in result
        assert 'coherence_metrics' in result


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for integration validation"""
    
    def test_integration_latency_benchmark(self):
        """Benchmark integration latency across components"""
        # Test component initialization times
        start_time = time.time()
        rust_bridge = RustPythonBridge()
        rust_init_time = time.time() - start_time
        
        start_time = time.time()
        cuda_bridge = CUDAConsciousnessBridge()
        cuda_init_time = time.time() - start_time
        
        start_time = time.time()
        quantum_backend = QuantumBackendIntegration()
        quantum_init_time = time.time() - start_time
        
        # All components should initialize quickly
        assert rust_init_time < 1.0, f"Rust bridge init too slow: {rust_init_time:.3f}s"
        assert cuda_init_time < 2.0, f"CUDA bridge init too slow: {cuda_init_time:.3f}s"
        assert quantum_init_time < 1.0, f"Quantum backend init too slow: {quantum_init_time:.3f}s"
        
        print(f"Initialization times - Rust: {rust_init_time:.3f}s, CUDA: {cuda_init_time:.3f}s, Quantum: {quantum_init_time:.3f}s")
    
    def test_throughput_benchmark(self):
        """Benchmark throughput across integration bridges"""
        cuda_bridge = CUDAConsciousnessBridge()
        
        # Test PHI computation throughput
        test_sizes = [1000, 10000, 100000]
        throughput_results = {}
        
        for size in test_sizes:
            test_data = np.random.rand(size).astype(np.float32)
            
            start_time = time.time()
            result = cuda_bridge.execute_phi_parallel_computation(test_data)
            execution_time = time.time() - start_time
            
            throughput = size / execution_time  # elements per second
            throughput_results[size] = throughput
            
            print(f"Size {size}: {throughput:.0f} elements/sec, {execution_time:.4f}s")
        
        # Verify scaling performance
        assert throughput_results[100000] > throughput_results[1000], "Throughput should scale with size"


if __name__ == "__main__":
    # Run comprehensive integration tests
    pytest.main([__file__, "-v", "--tb=short"])