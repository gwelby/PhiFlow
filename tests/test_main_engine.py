#!/usr/bin/env python3
"""
Test suite for PhiFlow Main Engine Integration
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

class TestPhiFlowQuantumConsciousnessEngine:
    """Test suite for the main PhiFlow engine"""
    
    def setup_method(self):
        """Setup for each test"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        # Initialize engine with default settings
        self.engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='simulator',
            enable_consciousness=True,
            enable_biofeedback=False
        )
    
    def test_initialization(self):
        """Test engine initialization"""
        assert self.engine.quantum_backend_type == 'simulator'
        assert self.engine.consciousness_enabled is True
        assert self.engine.biofeedback_enabled is False
        
        # Check component initialization
        assert self.engine.quantum_bridge is not None
        assert self.engine.consciousness_monitor is not None
        assert self.engine.consciousness_integrator is not None
        assert self.engine.coherence_engine is not None
        assert self.engine.quantum_optimizer is not None
    
    def test_performance_metrics_initialization(self):
        """Test performance metrics initialization"""
        metrics = self.engine.performance_metrics
        
        assert 'total_commands' in metrics
        assert 'quantum_commands' in metrics
        assert 'consciousness_optimized' in metrics
        assert 'average_coherence' in metrics
        assert 'average_phi_alignment' in metrics
        
        # Check initial values
        assert metrics['total_commands'] == 0
        assert metrics['quantum_commands'] == 0
        assert metrics['consciousness_optimized'] == 0
        assert metrics['average_coherence'] == 0.0
        assert metrics['average_phi_alignment'] == 0.0
    
    def test_execution_history_initialization(self):
        """Test execution history initialization"""
        assert isinstance(self.engine.execution_history, list)
        assert len(self.engine.execution_history) == 0
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.engine.get_system_status()
        
        assert 'engine_status' in status
        assert 'components' in status
        assert 'performance_metrics' in status
        assert 'execution_history_count' in status
        assert 'ready_for_phase_1' in status
        
        assert status['engine_status'] == 'initialized'
        assert status['ready_for_phase_1'] is True
        assert status['execution_history_count'] == 0
    
    def test_component_status_reporting(self):
        """Test individual component status reporting"""
        status = self.engine.get_system_status()
        components = status['components']
        
        assert 'quantum_bridge' in components
        assert 'consciousness_monitor' in components
        assert 'coherence_engine' in components
        assert 'quantum_optimizer' in components
        
        # Check that components are properly initialized
        assert components['quantum_bridge'] == 'ready'
        assert components['consciousness_monitor'] == 'ready'
        assert components['coherence_engine'] == 'stub'  # Phase 1 implementation
        assert components['quantum_optimizer'] == 'stub'  # Phase 1 implementation
    
    @pytest.mark.skip(reason="Implementation required in Phase 1")
    def test_execute_phiflow_program(self):
        """Test PhiFlow program execution"""
        program_source = """
        INITIALIZE frequency=432 coherence=1.0
        EVOLVE phi_level=2 frequency=528
        INTEGRATE compression=phi
        """
        
        result = self.engine.execute_phiflow_program(program_source)
        
        assert 'success' in result
        assert 'execution_time' in result
        assert 'quantum_results' in result
        assert 'consciousness_metrics' in result
        assert 'coherence_metrics' in result
    
    def test_execute_phiflow_program_stub(self):
        """Test PhiFlow program execution stub (Phase 0)"""
        program_source = "INITIALIZE frequency=432"
        
        result = self.engine.execute_phiflow_program(program_source)
        
        assert result['success'] is False
        assert result['message'] == 'Phase 1 implementation required'
        assert 'components_ready' in result
        
        components_ready = result['components_ready']
        assert components_ready['quantum_bridge'] is True
        assert components_ready['consciousness_monitor'] is True
        assert components_ready['coherence_engine'] is True
        assert components_ready['quantum_optimizer'] is True

class TestEngineInitializationVariations:
    """Test different engine initialization configurations"""
    
    def test_initialization_without_consciousness(self):
        """Test engine initialization without consciousness monitoring"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='simulator',
            enable_consciousness=False,
            enable_biofeedback=False
        )
        
        assert engine.consciousness_enabled is False
        assert engine.consciousness_monitor is None
        assert engine.consciousness_integrator is None
    
    def test_initialization_with_biofeedback(self):
        """Test engine initialization with biofeedback enabled"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='simulator',
            enable_consciousness=True,
            enable_biofeedback=True
        )
        
        assert engine.biofeedback_enabled is True
        assert engine.consciousness_monitor is not None
    
    def test_initialization_with_ibm_backend(self):
        """Test engine initialization with IBM quantum backend"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        # This should gracefully fall back to simulation if no token provided
        engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='ibm',
            ibm_token=None,
            enable_consciousness=True,
            enable_biofeedback=False
        )
        
        assert engine.quantum_backend_type == 'ibm'
        assert engine.quantum_bridge is not None

class TestEngineComponentIntegration:
    """Test integration between engine components"""
    
    def setup_method(self):
        """Setup for integration tests"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        self.engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='simulator',
            enable_consciousness=True,
            enable_biofeedback=False
        )
    
    def test_quantum_consciousness_integration(self):
        """Test quantum bridge and consciousness monitor integration"""
        # Both components should be available and working
        assert self.engine.quantum_bridge is not None
        assert self.engine.consciousness_monitor is not None
        assert self.engine.consciousness_integrator is not None
        
        # Test that consciousness integrator has reference to monitor
        assert self.engine.consciousness_integrator.monitor == self.engine.consciousness_monitor
    
    def test_coherence_engine_integration(self):
        """Test coherence engine integration with other components"""
        assert self.engine.coherence_engine is not None
        
        # Coherence engine should have references to other components
        assert self.engine.coherence_engine.quantum_bridge == self.engine.quantum_bridge
        assert self.engine.coherence_engine.consciousness_monitor == self.engine.consciousness_monitor
    
    def test_optimizer_integration(self):
        """Test quantum optimizer integration with consciousness monitoring"""
        assert self.engine.quantum_optimizer is not None
        
        # Optimizer should have reference to consciousness monitor
        assert self.engine.quantum_optimizer.consciousness_monitor == self.engine.consciousness_monitor

class TestEngineErrorHandling:
    """Test error handling and graceful degradation"""
    
    def test_missing_component_handling(self):
        """Test handling of missing components"""
        # This tests the graceful degradation when components are not available
        # The engine should still initialize but report component status correctly
        pass
    
    def test_invalid_backend_handling(self):
        """Test handling of invalid quantum backend"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        # Should gracefully handle invalid backend and fall back to simulation
        engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='invalid_backend',
            enable_consciousness=True,
            enable_biofeedback=False
        )
        
        assert engine.quantum_bridge is not None
        # Should fall back to phi_simulation mode

# Integration tests
@pytest.mark.integration
class TestFullSystemIntegration:
    """Full system integration tests"""
    
    def setup_method(self):
        """Setup for integration tests"""
        from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
        
        self.engine = PhiFlowQuantumConsciousnessEngine(
            quantum_backend='simulator',
            enable_consciousness=True,
            enable_biofeedback=False
        )
    
    @pytest.mark.skip(reason="Implementation required in Phase 1")
    def test_end_to_end_phiflow_execution(self):
        """Test complete end-to-end PhiFlow program execution"""
        program = """
        INITIALIZE frequency=432 coherence=1.0 purpose="Integration test"
        TRANSITION phi_level=1 frequency=528
        EVOLVE phi_level=2 frequency=594
        INTEGRATE compression=phi frequency=672
        """
        
        result = self.engine.execute_phiflow_program(program)
        
        assert result['success'] is True
        assert result['quantum_results'] is not None
        assert result['consciousness_metrics'] is not None
        assert result['coherence_metrics'] is not None
        assert result['performance_metrics'] is not None
    
    def test_system_health_check(self):
        """Test comprehensive system health check"""
        status = self.engine.get_system_status()
        
        # All critical components should be available
        components = status['components']
        critical_components = ['quantum_bridge', 'consciousness_monitor']
        
        for component in critical_components:
            assert components[component] == 'ready'
        
        # Phase 1 components should be in stub state
        phase1_components = ['coherence_engine', 'quantum_optimizer']
        
        for component in phase1_components:
            assert components[component] == 'stub'

if __name__ == "__main__":
    pytest.main([__file__, "-v"])