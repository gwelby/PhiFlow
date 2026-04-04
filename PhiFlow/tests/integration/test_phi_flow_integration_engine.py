"""
Test Suite for PhiFlow Integration Engine

Comprehensive tests for:
- Component initialization
- 8-phase execution pipeline
- Real-time monitoring
- Performance metrics
- Consciousness optimization
- Error handling and recovery
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../PhiFlow/src'))

from integration.phi_flow_integration_engine import (
    PhiFlowIntegrationEngine,
    OptimizationLevel,
    ConsciousnessState,
    ExecutionPhase,
    ExecutionMetrics,
    HealthStatus,
    PHI,
    LAMBDA,
    PHI_PHI,
    SACRED_FREQUENCIES
)

class TestPhiFlowIntegrationEngine:
    """Test suite for PhiFlowIntegrationEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create a test engine instance"""
        return PhiFlowIntegrationEngine(
            enable_cuda=False,  # Disable CUDA for testing
            debug=True,
            monitoring_frequency_hz=20.0  # Higher frequency for faster tests
        )
    
    @pytest.fixture
    def mock_consciousness_monitor(self):
        """Create a mock consciousness monitor"""
        monitor = Mock()
        monitor.get_state.return_value = ConsciousnessState.TRANSCEND
        return monitor
    
    def test_engine_initialization(self, engine):
        """Test that engine initializes correctly"""
        assert engine.components_initialized is True
        assert engine.initialization_error is None
        assert engine.monitoring_active is True
        assert hasattr(engine, 'coherence_engine')
        assert hasattr(engine, 'optimizer')
        assert hasattr(engine, 'lexer')
        assert hasattr(engine, 'parser')
        assert hasattr(engine, 'semantic_analyzer')
        assert hasattr(engine, 'compiler')
    
    def test_sacred_mathematics_constants(self, engine):
        """Test that sacred mathematics constants are correct"""
        assert engine.phi_cache['phi'] == PHI
        assert engine.phi_cache['lambda'] == LAMBDA
        assert engine.phi_cache['phi_phi'] == PHI_PHI
        assert abs(PHI - 1.618033988749895) < 1e-10
        assert abs(PHI_PHI - 11.09017095324081) < 1e-10
    
    def test_sacred_frequencies(self, engine):
        """Test sacred frequency values"""
        frequencies = engine.phi_cache['frequencies']
        assert frequencies['ground'] == 432
        assert frequencies['creation'] == 528
        assert frequencies['heart'] == 594
        assert frequencies['voice'] == 672
        assert frequencies['vision'] == 720
        assert frequencies['unity'] == 768
        assert frequencies['source'] == 963
    
    def test_health_status(self, engine):
        """Test system health status reporting"""
        health = engine.get_health_status()
        
        assert isinstance(health, HealthStatus)
        assert 0.0 <= health.overall_health <= 1.0
        assert isinstance(health.coherence_engine_status, bool)
        assert isinstance(health.optimizer_status, bool)
        assert isinstance(health.parser_status, bool)
        assert health.memory_available_gb > 0
        assert 0 <= health.cpu_usage_percent <= 100
        assert isinstance(health.timestamp, datetime)
    
    def test_real_time_monitoring(self, engine):
        """Test real-time monitoring system"""
        # Allow monitoring to run for a short time
        time.sleep(0.2)
        
        assert engine.monitoring_active is True
        assert engine.monitoring_thread is not None
        assert engine.monitoring_thread.is_alive()
        assert 0.0 <= engine.current_coherence <= 1.0
        assert isinstance(engine.current_consciousness_state, ConsciousnessState)
    
    def test_coherence_measurement(self, engine):
        """Test coherence measurement and correction"""
        initial_coherence = engine.coherence_engine.measure_coherence()
        assert 0.0 <= initial_coherence <= 1.0
        
        # Test coherence correction
        target_coherence = 0.999
        corrected = engine.coherence_engine.apply_correction(target_coherence)
        assert corrected >= initial_coherence
        assert corrected <= 1.0
    
    def test_optimization_levels(self, engine):
        """Test all optimization levels"""
        expected_speedups = {
            OptimizationLevel.LINEAR: 1.0,
            OptimizationLevel.FIBONACCI: PHI,
            OptimizationLevel.PARALLEL: PHI ** 2,
            OptimizationLevel.QUANTUM_LIKE: PHI ** 3,
            OptimizationLevel.CONSCIOUSNESS: PHI ** 4,
            OptimizationLevel.CONSCIOUSNESS_QUANTUM: PHI_PHI
        }
        
        for level, expected_speedup in expected_speedups.items():
            result = engine.optimizer.optimize({"mock": "ast"}, level)
            assert abs(result['speedup'] - expected_speedup) < 1e-10
            assert result['optimization_level'] == level
            assert 0.0 <= result['phi_efficiency'] <= 1.0
    
    def test_simple_program_execution(self, engine):
        """Test execution of a simple PhiFlow program"""
        simple_program = """
        phi_program test() {
            frequency f = 432.0;
            execute();
        }
        """
        
        result = engine.execute_program(
            source_code=simple_program,
            optimization_level=OptimizationLevel.FIBONACCI
        )
        
        # Verify successful execution
        assert result['success'] is True
        assert 'execution_id' in result
        assert 'output' in result
        assert 'metrics' in result
        assert 'performance' in result
        assert 'phases' in result
        
        # Verify metrics
        metrics = result['metrics']
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.success is True
        assert metrics.total_duration > 0
        assert metrics.tokens_processed > 0
        assert metrics.speedup_achieved == PHI  # Fibonacci level
        
        # Verify all phases completed
        assert len(result['phases']) == len(ExecutionPhase)
        for phase in ExecutionPhase:
            assert phase.value in result['phases']
            assert result['phases'][phase.value] >= 0
    
    def test_complex_program_execution(self, engine):
        """Test execution of a complex PhiFlow program"""
        complex_program = """
        phi_program complex_test() {
            frequency ground = 432.0;
            frequency creation = 528.0;
            phi_level optimization = φ^φ;
            
            consciousness_state state = TRANSCEND;
            coherence_target = 0.999;
            
            for i in range(fibonacci(8)) {
                process_with_phi(i * φ);
                maintain_coherence(coherence_target);
            }
            
            execute_with_sacred_geometry();
        }
        """
        
        result = engine.execute_program(
            source_code=complex_program,
            optimization_level=OptimizationLevel.CONSCIOUSNESS_QUANTUM
        )
        
        assert result['success'] is True
        assert result['performance']['speedup_achieved'] == PHI_PHI
        assert result['performance']['coherence_maintained'] >= 0.95
        assert result['performance']['consciousness_enhancement'] >= 1.8
        assert result['performance']['frequency_alignment'] >= 0.90
    
    def test_consciousness_state_optimization(self, engine):
        """Test consciousness state optimization"""
        for state in ConsciousnessState:
            result = engine.optimize_consciousness_state(state)
            
            assert result['target_state'] == state.value
            assert result['target_frequency_hz'] > 0
            assert result['phi_tuning_factor'] > 0
            assert result['optimization_bonus'] >= 1.0
            assert 0.0 <= result['coherence_before'] <= 1.0
            assert 0.0 <= result['coherence_after'] <= 1.0
            assert result['coherence_after'] >= result['coherence_before']
            assert 0.0 <= result['frequency_alignment'] <= 1.0
            
            # Verify state was updated
            assert engine.current_consciousness_state == state
    
    def test_error_handling(self, engine):
        """Test error handling in program execution"""
        # Test with invalid source code
        invalid_program = "this is not valid PhiFlow code !!!"
        
        result = engine.execute_program(invalid_program)
        
        # Execution should handle errors gracefully
        assert 'execution_id' in result
        assert 'metrics' in result
        
        # Check that metrics were recorded even for failed execution
        metrics = result['metrics']
        assert isinstance(metrics, ExecutionMetrics)
        assert metrics.execution_id is not None
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.total_duration > 0
    
    def test_performance_analytics(self, engine):
        """Test performance analytics collection"""
        # Execute several programs to build history
        programs = [
            "simple test program 1",
            "simple test program 2", 
            "simple test program 3"
        ]
        
        for i, program in enumerate(programs):
            engine.execute_program(
                program,
                optimization_level=OptimizationLevel.FIBONACCI
            )
        
        analytics = engine.get_performance_analytics()
        
        assert analytics['total_executions'] >= 3
        assert analytics['successful_executions'] >= 0
        assert 0.0 <= analytics['success_rate'] <= 1.0
        assert 'averages' in analytics
        assert 'phase_performance' in analytics
        assert 'recent_executions' in analytics
        
        # Verify averages
        averages = analytics['averages']
        assert averages['duration_seconds'] >= 0
        assert averages['speedup_achieved'] >= 1.0
        assert 0.0 <= averages['coherence_maintained'] <= 1.0
        
        # Verify phase performance
        phase_perf = analytics['phase_performance']
        for phase in ExecutionPhase:
            if phase.value in phase_perf:
                assert phase_perf[phase.value] >= 0
    
    def test_execution_history(self, engine):
        """Test execution history tracking"""
        initial_count = len(engine.execution_history)
        
        # Execute a program
        engine.execute_program("test program for history")
        
        # Verify history was updated
        assert len(engine.execution_history) == initial_count + 1
        
        # Verify latest execution
        latest = engine.execution_history[-1]
        assert isinstance(latest, ExecutionMetrics)
        assert latest.execution_id is not None
        assert latest.start_time is not None
        assert latest.total_duration is not None
    
    def test_phi_efficiency_calculation(self, engine):
        """Test phi efficiency calculations"""
        # Test with different optimization levels
        for level in OptimizationLevel:
            result = engine.execute_program(
                "test program",
                optimization_level=level
            )
            
            phi_efficiency = result['performance']['phi_efficiency']
            assert 0.0 <= phi_efficiency <= 1.0
            
            # Higher optimization levels should have higher phi efficiency
            if level == OptimizationLevel.CONSCIOUSNESS_QUANTUM:
                assert phi_efficiency == 1.0  # Maximum efficiency
    
    def test_cuda_integration(self):
        """Test CUDA integration (when enabled)"""
        # Test with CUDA enabled
        cuda_engine = PhiFlowIntegrationEngine(
            enable_cuda=True,
            debug=True
        )
        
        try:
            result = cuda_engine.execute_program("test cuda program")
            
            # CUDA should provide acceleration
            if 'cuda_acceleration' in result:
                assert result['cuda_acceleration'] >= 1.0
                assert result['memory_usage_gb'] > 0
        finally:
            cuda_engine.shutdown()
    
    def test_frequency_alignment(self, engine):
        """Test sacred frequency alignment calculations"""
        # Test alignment for each consciousness state
        for state in ConsciousnessState:
            engine.current_consciousness_state = state
            alignment = engine._calculate_frequency_alignment()
            
            assert 0.0 <= alignment <= 1.0
            assert alignment >= 0.90  # Should maintain high alignment
    
    def test_consciousness_enhancement(self, engine):
        """Test consciousness enhancement calculations"""
        enhancement = engine._calculate_consciousness_enhancement()
        
        assert enhancement >= 1.0  # Should enhance performance
        assert enhancement >= 1.8   # Base enhancement from consciousness expert
        assert enhancement <= 5.0   # Reasonable upper bound
    
    def test_monitoring_frequency(self):
        """Test different monitoring frequencies"""
        frequencies = [1.0, 5.0, 10.0, 20.0]
        
        for freq in frequencies:
            engine = PhiFlowIntegrationEngine(
                monitoring_frequency_hz=freq,
                debug=True
            )
            
            try:
                assert engine.monitoring_frequency_hz == freq
                assert engine.monitoring_active is True
                
                # Allow some monitoring cycles
                time.sleep(0.3)
                
                # Verify monitoring is working
                assert engine.current_coherence > 0
                
            finally:
                engine.shutdown()
    
    def test_graceful_shutdown(self, engine):
        """Test graceful shutdown"""
        assert engine.monitoring_active is True
        
        engine.shutdown()
        
        assert engine.monitoring_active is False
        
        # Allow time for thread to finish
        time.sleep(0.1)
        
        if engine.monitoring_thread:
            assert not engine.monitoring_thread.is_alive()
    
    def test_execution_metrics_completeness(self, engine):
        """Test that execution metrics are comprehensive"""
        result = engine.execute_program("comprehensive metrics test")
        
        metrics = result['metrics']
        
        # Verify all important metrics are present
        assert metrics.execution_id is not None
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.total_duration is not None
        assert metrics.tokens_processed is not None
        assert metrics.ast_nodes_generated is not None
        assert metrics.bytecode_instructions is not None
        assert metrics.speedup_achieved is not None
        assert metrics.phi_efficiency is not None
        assert metrics.coherence_start is not None
        assert metrics.coherence_end is not None
        assert metrics.coherence_average is not None
        assert metrics.consciousness_enhancement is not None
        assert metrics.frequency_alignment is not None
        assert metrics.memory_peak_mb is not None
        assert metrics.cpu_usage_percent is not None
        
        # Verify phase timings
        assert len(metrics.phase_timings) == len(ExecutionPhase)
        for phase in ExecutionPhase:
            assert phase in metrics.phase_timings
            assert metrics.phase_timings[phase] >= 0
    
    def test_integration_with_consciousness_monitor(self):
        """Test integration with consciousness monitor"""
        mock_monitor = Mock()
        mock_monitor.get_state.return_value = ConsciousnessState.SUPERPOSITION
        
        engine = PhiFlowIntegrationEngine(
            consciousness_monitor=mock_monitor,
            monitoring_frequency_hz=10.0,
            debug=True
        )
        
        try:
            # Allow monitoring to update state
            time.sleep(0.2)
            
            # Verify consciousness monitor is being called
            assert mock_monitor.get_state.called
            
            # Verify state was updated from monitor
            assert engine.current_consciousness_state == ConsciousnessState.SUPERPOSITION
            
        finally:
            engine.shutdown()

class TestExecutionMetrics:
    """Test suite for ExecutionMetrics data class"""
    
    def test_metrics_initialization(self):
        """Test ExecutionMetrics initialization"""
        metrics = ExecutionMetrics(
            execution_id="test_123",
            start_time=datetime.now()
        )
        
        assert metrics.execution_id == "test_123"
        assert isinstance(metrics.start_time, datetime)
        assert metrics.end_time is None
        assert metrics.total_duration is None
        assert len(metrics.phase_timings) == 0
        assert len(metrics.coherence_samples) == 0
        assert len(metrics.warnings) == 0
        assert metrics.success is False

class TestHealthStatus:
    """Test suite for HealthStatus data class"""
    
    def test_health_status_initialization(self):
        """Test HealthStatus initialization"""
        health = HealthStatus(
            overall_health=0.95,
            coherence_engine_status=True,
            optimizer_status=True,
            parser_status=True,
            consciousness_monitor_status=True,
            cuda_status=False,
            memory_available_gb=16.0,
            cpu_usage_percent=35.0
        )
        
        assert health.overall_health == 0.95
        assert health.coherence_engine_status is True
        assert health.cuda_status is False
        assert health.memory_available_gb == 16.0
        assert health.cpu_usage_percent == 35.0
        assert isinstance(health.timestamp, datetime)

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])