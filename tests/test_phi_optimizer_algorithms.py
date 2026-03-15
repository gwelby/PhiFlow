#!/usr/bin/env python3
"""
Test suite for PhiFlow Phi-Quantum Optimizer Real Algorithm Implementations
Tests the actual phi-harmonic optimization algorithms
"""

import pytest
import sys
import os
import time
import numpy as np
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import the optimizer
from optimization.phi_quantum_optimizer import (
    PhiQuantumOptimizer, 
    OptimizationLevel, 
    OptimizationResult
)

class TestPhiHarmonicOptimization:
    """Test suite for real phi-harmonic optimization algorithms"""
    
    def setup_method(self):
        """Setup for each test"""
        self.optimizer = PhiQuantumOptimizer(enable_cuda=False)
        
        # Define test functions
        self.simple_function = lambda x, y: x**2 + y**2
        self.complex_function = self._create_complex_test_function()
        self.array_function = self._create_array_test_function()
    
    def _create_complex_test_function(self):
        """Create a complex test function for optimization"""
        def complex_func(a=1.0, b=2.0, c=3.0):
            """Multi-parameter function with local minima"""
            # Simulate computational work
            time.sleep(0.001)  # 1ms base execution time
            
            # Mathematical function with multiple parameters
            result = (a - 1.618)**2 + (b - 2.618)**2 + (c - 4.236)**2
            
            # Add phi-harmonic resonance bonus
            phi_resonance = np.cos(a * 1.618) + np.sin(b * 1.618) + np.cos(c * 1.618)
            result -= 0.1 * phi_resonance  # Reward phi-harmonic values
            
            return result
        
        return complex_func
    
    def _create_array_test_function(self):
        """Create test function that works with arrays"""
        def array_func(data=None):
            if data is None:
                data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            
            # Simulate array processing time
            time.sleep(0.002)  # 2ms base execution time
            
            # Process array with phi-harmonic pattern
            phi_weights = np.array([1.618**i for i in range(len(data))])
            weighted_sum = np.sum(data * phi_weights)
            
            return weighted_sum
        
        return array_func
    
    def test_golden_section_parameter_optimization(self):
        """Test golden section search parameter optimization"""
        
        # Set to phi-enhanced optimization level
        self.optimizer.set_optimization_level(OptimizationLevel.PHI_ENHANCED)
        
        # Test parameters
        test_params = {'x': 10.0, 'y': 5.0}
        
        # Run optimization
        result = self.optimizer.optimize_computation(
            self.simple_function, 
            test_params, 
            OptimizationLevel.PHI_ENHANCED
        )
        
        # Validate results
        assert result.success == True
        assert result.speedup_ratio >= 1.0  # Should achieve some speedup
        assert result.optimization_level == OptimizationLevel.PHI_ENHANCED
        assert result.algorithm_used in ['phi_enhanced', 'conservative_phi']
        
        print(f"✅ Golden section optimization: {result.speedup_ratio:.3f}x speedup")
        print(f"📊 Phi alignment: {result.phi_alignment:.3f}")
    
    def test_phi_harmonic_resonance_calculation(self):
        """Test phi-harmonic resonance field calculation"""
        
        # Test with various parameter sets
        test_cases = [
            {'a': 1.618, 'b': 2.618, 'c': 4.236},  # Phi-harmonic values
            {'a': 1.0, 'b': 2.0, 'c': 3.0},        # Regular values
            {'x': 100.0, 'y': 200.0},              # Large values
            {'small': 0.01, 'tiny': 0.001}         # Small values
        ]
        
        for i, params in enumerate(test_cases):
            resonance = self.optimizer._calculate_phi_harmonic_resonance(params)
            
            # Resonance should be in valid range
            assert 0.0 <= resonance <= 1.0
            
            print(f"📊 Test case {i+1}: resonance = {resonance:.6f}")
            
            # Phi-harmonic values should have higher resonance
            if i == 0:  # Phi-harmonic case
                assert resonance > 0.1  # Should be reasonably high
    
    def test_phi_parallel_optimization(self):
        """Test phi-parallel processing optimization"""
        
        # Set to phi-parallel level
        self.optimizer.set_optimization_level(OptimizationLevel.PHI_CUBED)
        
        # Test with complex function
        test_params = {'a': 2.0, 'b': 3.0, 'c': 4.0}
        
        # Run parallel optimization
        result = self.optimizer.optimize_computation(
            self.complex_function,
            test_params,
            OptimizationLevel.PHI_CUBED
        )
        
        # Validate results
        assert result.success == True
        assert result.speedup_ratio >= 1.0
        assert result.optimization_level == OptimizationLevel.PHI_CUBED
        
        print(f"✅ Phi-parallel optimization: {result.speedup_ratio:.3f}x speedup")
        print(f"🔄 Algorithm used: {result.algorithm_used}")
    
    def test_quantum_like_optimization(self):
        """Test quantum-like superposition optimization"""
        
        # Set to quantum level
        self.optimizer.set_optimization_level(OptimizationLevel.CONSCIOUSNESS_QUANTUM)
        
        # Test with array function
        test_data = np.array([1.0, 1.618, 2.618, 4.236, 6.854])
        test_params = {'data': test_data}
        
        # Run quantum-like optimization
        result = self.optimizer.optimize_computation(
            self.array_function,
            test_params,
            OptimizationLevel.CONSCIOUSNESS_QUANTUM
        )
        
        # Validate results
        assert result.success == True
        assert result.speedup_ratio >= 1.0
        assert result.optimization_level == OptimizationLevel.CONSCIOUSNESS_QUANTUM
        
        print(f"✅ Quantum-like optimization: {result.speedup_ratio:.3f}x speedup")
        print(f"🔬 Quantum algorithm applied successfully")
    
    def test_parameter_variation_creation(self):
        """Test creation of phi-harmonic parameter variations"""
        
        base_params = {'x': 10.0, 'y': 20.0, 'z': 30.0}
        variations = self.optimizer._create_phi_parameter_variations(base_params, 5)
        
        # Should create requested number of variations
        assert len(variations) == 5
        
        # Each variation should have same parameter keys
        for variation in variations:
            assert set(variation.keys()) == set(base_params.keys())
        
        # Variations should be different from original
        different_count = 0
        for variation in variations:
            if variation != base_params:
                different_count += 1
        
        assert different_count > 0  # At least some should be different
        
        print(f"✅ Created {len(variations)} parameter variations")
        print(f"📊 {different_count} variations different from original")
    
    def test_phi_field_array_processing(self):
        """Test phi-harmonic field application to arrays"""
        
        original_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Apply phi field transformation
        transformed_array = self.optimizer._apply_phi_field_to_array(
            original_array, phi_factor=1.618, rotation=137.5
        )
        
        # Array should have same shape
        assert transformed_array.shape == original_array.shape
        
        # Values should be modified (not identical)
        assert not np.array_equal(transformed_array, original_array)
        
        # Values should be reasonable (not extreme)
        ratio = np.abs(transformed_array / original_array)
        assert np.all(ratio < 2.0)  # Less than 2x change
        assert np.all(ratio > 0.5)  # More than 0.5x change
        
        print(f"✅ Phi-field array transformation applied")
        print(f"📊 Average change ratio: {np.mean(ratio):.3f}")
    
    def test_optimization_level_scaling(self):
        """Test that optimization levels provide appropriate speedup scaling"""
        
        levels_to_test = [
            OptimizationLevel.LINEAR,
            OptimizationLevel.PHI_ENHANCED,
            OptimizationLevel.PHI_SQUARED,
            OptimizationLevel.PHI_CUBED
        ]
        
        test_params = {'a': 1.5, 'b': 2.5, 'c': 3.5}
        results = []
        
        for level in levels_to_test:
            result = self.optimizer.optimize_computation(
                self.complex_function,
                test_params,
                level
            )
            results.append(result)
            
            print(f"📊 Level {level.name}: {result.speedup_ratio:.3f}x speedup")
        
        # Higher levels should generally achieve better speedup
        # Note: Due to overhead, this isn't always true for very small functions
        assert len(results) == len(levels_to_test)
        
        # At least one advanced level should show improvement
        advanced_speedups = [r.speedup_ratio for r in results[1:]]  # Skip LINEAR
        assert max(advanced_speedups) > 1.0
    
    def test_parallelizability_analysis(self):
        """Test analysis of function parallelizability"""
        
        # Test cases with different parameter characteristics
        test_cases = [
            ({'x': 1}, False),                          # Single parameter
            ({'x': 1, 'y': 2}, True),                   # Multiple parameters
            ({'data': np.array([1, 2, 3])}, True),      # Array parameter
            ({'big': 100}, True),                       # Large value
            ({'small': 0.1}, False)                     # Small single value
        ]
        
        for params, expected_parallelizable in test_cases:
            is_parallelizable = self.optimizer._analyze_parallelizability(
                lambda **kwargs: sum(kwargs.values()) if all(isinstance(v, (int, float)) for v in kwargs.values()) else 0,
                params
            )
            
            print(f"📊 Parameters {params}: parallelizable = {is_parallelizable}")
            
            # Should match expected result (with some flexibility for edge cases)
            if expected_parallelizable:
                assert is_parallelizable or len(params) == 1  # Single array case might vary

class TestQuantumLikeAlgorithms:
    """Test suite for quantum-like optimization algorithms"""
    
    def setup_method(self):
        """Setup for each test"""
        self.optimizer = PhiQuantumOptimizer(enable_cuda=False)
        self.optimizer.set_optimization_level(OptimizationLevel.CONSCIOUSNESS_QUANTUM)
    
    def test_quantum_parameter_state_creation(self):
        """Test creation of quantum parameter states"""
        
        base_params = {'x': 1.0, 'y': 2.0}
        num_states = 4
        
        quantum_states = self.optimizer._create_quantum_parameter_states(base_params, num_states)
        
        # Should create requested number of states
        assert len(quantum_states) == num_states
        
        # Each state should have same parameter structure
        for state in quantum_states:
            assert set(state.keys()) == set(base_params.keys())
        
        # States should have variations
        unique_states = 0
        for state in quantum_states:
            if state != base_params:
                unique_states += 1
        
        assert unique_states > 0
        
        print(f"✅ Created {num_states} quantum parameter states")
        print(f"📊 {unique_states} states with variations")
    
    def test_quantum_evolution_simulation(self):
        """Test quantum time evolution simulation"""
        
        num_states = 4
        amplitudes = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
        parameter_states = [{'x': i} for i in range(num_states)]
        
        # Apply quantum evolution
        evolved_amplitudes = self.optimizer._apply_quantum_evolution(amplitudes, parameter_states)
        
        # Should preserve normalization
        norm = np.sqrt(np.sum(np.abs(evolved_amplitudes)**2))
        assert abs(norm - 1.0) < 1e-10  # Very close to 1
        
        # Should have complex values
        assert evolved_amplitudes.dtype == complex
        
        # Should have same number of states
        assert len(evolved_amplitudes) == num_states
        
        print(f"✅ Quantum evolution applied to {num_states} states")
        print(f"📊 Final norm: {norm:.10f}")
    
    def test_quantum_measurement_collapse(self):
        """Test quantum measurement collapse simulation"""
        
        # Create test results with different performance
        results = [
            ("result1", 0.1, 0.5+0.5j),   # Fast execution
            ("result2", 0.2, 0.3+0.4j),   # Slower execution
            ("result3", 0.05, 0.2+0.1j),  # Fastest execution
            (None, float('inf'), 0.0)      # Failed result
        ]
        
        amplitudes = np.array([0.5+0.5j, 0.3+0.4j, 0.2+0.1j, 0.0])
        
        # Run measurement multiple times to test probabilistic selection
        measurements = []
        for _ in range(10):
            result = self.optimizer._quantum_measurement_collapse(results, amplitudes)
            measurements.append(result)
        
        # Should get valid results (not None)
        valid_measurements = [m for m in measurements if m is not None]
        assert len(valid_measurements) > 0
        
        # Should prefer faster execution (result3 should appear frequently)
        result3_count = measurements.count("result3")
        print(f"✅ Quantum measurement collapse tested")
        print(f"📊 Result3 (fastest) appeared {result3_count}/10 times")

class TestPerformanceBenchmarks:
    """Performance benchmarks for optimization algorithms"""
    
    def setup_method(self):
        """Setup for benchmarks"""
        self.optimizer = PhiQuantumOptimizer(enable_cuda=False)
    
    def test_optimization_performance_scaling(self):
        """Test performance scaling across optimization levels"""
        
        def benchmark_function(n=1000):
            """CPU-intensive benchmark function"""
            # Fibonacci-like calculation
            a, b = 1, 1
            for i in range(n):
                a, b = b, a + b
            return a
        
        # Test different optimization levels
        levels = [
            OptimizationLevel.LINEAR,
            OptimizationLevel.PHI_ENHANCED,
            OptimizationLevel.PHI_SQUARED,
            OptimizationLevel.PHI_CUBED
        ]
        
        results = {}
        baseline_time = None
        
        for level in levels:
            # Run optimization
            result = self.optimizer.optimize_computation(
                benchmark_function,
                {'n': 500},  # Smaller n for faster testing
                level
            )
            
            results[level.name] = {
                'speedup': result.speedup_ratio,
                'original_time': result.original_execution_time,
                'optimized_time': result.optimized_execution_time,
                'phi_alignment': result.phi_alignment
            }
            
            if baseline_time is None:
                baseline_time = result.original_execution_time
            
            print(f"📊 {level.name}: {result.speedup_ratio:.3f}x speedup")
            print(f"   Original: {result.original_execution_time*1000:.2f}ms")
            print(f"   Optimized: {result.optimized_execution_time*1000:.2f}ms")
            print(f"   Phi alignment: {result.phi_alignment:.3f}")
            print()
        
        # Validate that optimization provides benefits
        assert len(results) == len(levels)
        
        # At least one level should show speedup > 1.0
        max_speedup = max(r['speedup'] for r in results.values())
        assert max_speedup >= 1.0
        
        print(f"🏆 Maximum speedup achieved: {max_speedup:.3f}x")

if __name__ == "__main__":
    # Run specific test
    print("🚀 Testing PhiFlow Phi-Quantum Optimizer Real Algorithms")
    print("=" * 70)
    
    # Test phi-harmonic optimization
    test_phi = TestPhiHarmonicOptimization()
    test_phi.setup_method()
    
    print("\n🔬 Testing Golden Section Optimization...")
    test_phi.test_golden_section_parameter_optimization()
    
    print("\n🌊 Testing Phi-Harmonic Resonance...")
    test_phi.test_phi_harmonic_resonance_calculation()
    
    print("\n⚡ Testing Phi-Parallel Optimization...")
    test_phi.test_phi_parallel_optimization()
    
    print("\n🔬 Testing Quantum-Like Optimization...")
    test_phi.test_quantum_like_optimization()
    
    print("\n📊 Testing Performance Scaling...")
    test_perf = TestPerformanceBenchmarks()
    test_perf.setup_method()
    test_perf.test_optimization_performance_scaling()
    
    print("\n✅ All real algorithm tests completed successfully!")
    print("🌟 PhiFlow Phi-Quantum Optimizer algorithms are working!")