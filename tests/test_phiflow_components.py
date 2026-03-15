#!/usr/bin/env python3
"""
Comprehensive test framework for PhiFlow components
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

class TestQuantumBridge:
    """Test suite for PhiQuantumBridge"""
    
    def setup_method(self):
        """Setup for each test"""
        sys.path.insert(0, os.path.join('src', 'quantum_bridge'))
        from phi_quantum_interface import PhiQuantumBridge
        self.bridge = PhiQuantumBridge('simulator')
    
    def test_initialization(self):
        """Test quantum bridge initialization"""
        assert self.bridge.backend_type in ['simulator', 'phi_simulation']
        assert self.bridge.phi_coherence == 1.0
        assert self.bridge.consciousness_state == "OBSERVE"
    
    def test_phiflow_command_execution(self):
        """Test PhiFlow command execution"""
        result = self.bridge.execute_phiflow_command(
            'INITIALIZE', 432, {'coherence': 1.0}
        )
        
        assert result['execution_success'] is True
        assert 'phi_coherence' in result
        assert 'phi_resonance' in result
        assert 0 <= result['phi_coherence'] <= 1
        assert 0 <= result['phi_resonance'] <= 1
    
    def test_sacred_frequencies(self):
        """Test all sacred frequencies"""
        sacred_freqs = [432, 528, 594, 672, 720, 768, 963]
        
        for freq in sacred_freqs:
            result = self.bridge.execute_phiflow_command(
                'EVOLVE', freq, {'phi_level': 2}
            )
            assert result['execution_success'] is True
            assert result['phi_coherence'] >= 0
    
    def test_quantum_status(self):
        """Test quantum status reporting"""
        status = self.bridge.get_quantum_status()
        
        assert 'backend_type' in status
        assert 'phi_coherence' in status
        assert 'supported_frequencies' in status
        assert len(status['supported_frequencies']) == 7

class TestConsciousnessInterface:
    """Test suite for consciousness interface"""
    
    def setup_method(self):
        """Setup for each test"""
        sys.path.insert(0, os.path.join('src', 'consciousness'))
        from phi_consciousness_interface import ConsciousnessMonitor, PhiConsciousnessIntegrator
        self.monitor = ConsciousnessMonitor(enable_biofeedback=False)
        self.integrator = PhiConsciousnessIntegrator(self.monitor)
    
    def test_consciousness_measurement(self):
        """Test consciousness state measurement"""
        state = self.monitor.measure_consciousness_state()
        
        assert state.state_name in ["OBSERVE", "CREATE", "INTEGRATE", "HARMONIZE", "TRANSCEND", "CASCADE", "SUPERPOSITION"]
        assert 0 <= state.heart_coherence <= 1
        assert 0 <= state.phi_alignment <= 1
        assert 1 <= state.awareness_level <= 12
        assert state.frequency in [432, 528, 594, 672, 720, 768, 963]
    
    def test_consciousness_optimization(self):
        """Test consciousness-based optimization"""
        result = self.integrator.optimize_phi_command_for_consciousness(
            'EVOLVE', 528, {'coherence': 0.9, 'phi_level': 3}
        )
        
        assert 'optimized_parameters' in result
        assert 'resonance_score' in result
        assert 'consciousness_state' in result
        assert 0 <= result['resonance_score'] <= 1
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop"""
        assert not self.monitor.is_monitoring
        
        self.monitor.start_monitoring()
        assert self.monitor.is_monitoring
        
        self.monitor.stop_monitoring()
        assert not self.monitor.is_monitoring

class TestPhiMathematics:
    """Test suite for phi-harmonic mathematics"""
    
    def test_phi_constant(self):
        """Test phi constant accuracy"""
        PHI = 1.618033988749895
        assert abs(PHI - (1 + np.sqrt(5)) / 2) < 1e-10
    
    def test_golden_angle(self):
        """Test golden angle calculation"""
        GOLDEN_ANGLE = 137.5077640
        expected = 360 * (1 - 1/((1 + np.sqrt(5)) / 2))
        assert abs(GOLDEN_ANGLE - expected) < 0.001
    
    def test_sacred_frequencies(self):
        """Test sacred frequency relationships"""
        sacred_freqs = [432, 528, 594, 672, 720, 768, 963]
        
        # Test that frequencies follow harmonic relationships
        for i in range(len(sacred_freqs) - 1):
            ratio = sacred_freqs[i + 1] / sacred_freqs[i]
            assert 1.0 < ratio < 2.0  # Reasonable harmonic ratio

class TestSystemIntegration:
    """Integration tests for complete system"""
    
    def setup_method(self):
        """Setup integrated system"""
        # We'll implement this after fixing the main engine imports
        pass
    
    def test_end_to_end_execution(self):
        """Test complete PhiFlow program execution"""
        # This will be implemented in Phase 1
        pytest.skip("Requires complete integration engine")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        # This will be implemented in Phase 1
        pytest.skip("Requires performance monitoring system")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])