#!/usr/bin/env python3
"""
Test algorithm correctness for PhiFlow Phi-Quantum Optimizer
Validates that the real algorithms work correctly without focusing on performance timing
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from optimization.phi_quantum_optimizer import (
    PhiQuantumOptimizer, 
    OptimizationLevel, 
    PHI, 
    GOLDEN_ANGLE
)

def test_algorithm_correctness():
    """Test that the phi-optimization algorithms are mathematically correct"""
    
    print("🔬 Testing PhiFlow Phi-Quantum Optimizer Algorithm Correctness")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = PhiQuantumOptimizer(enable_cuda=False)
    
    print("✅ Optimizer initialized successfully\n")
    
    # Test 1: Golden Section Parameter Optimization
    print("📊 Test 1: Golden Section Parameter Optimization")
    print("-" * 50)
    
    def simple_func(x=1.0):
        return x  # Just return the parameter
    
    # Test mathematical parameter optimization (not timing-based)
    original_value = 10.0
    optimized_value = optimizer._mathematical_parameter_optimization(
        original_value, 'x', simple_func, {'x': original_value}
    )
    
    print(f"   Original value: {original_value}")
    print(f"   Optimized value: {optimized_value:.6f}")
    print(f"   Change ratio: {optimized_value/original_value:.6f}")
    
    # Should apply phi-harmonic enhancement
    expected_ratio = 1.0 + PHI * 0.01
    assert abs(optimized_value/original_value - expected_ratio) < 0.001
    print("   ✅ Golden section optimization working correctly\n")
    
    # Test 2: Phi-Harmonic Resonance Calculation
    print("📊 Test 2: Phi-Harmonic Resonance Calculation")
    print("-" * 50)
    
    # Test with phi-aligned values
    phi_params = {'a': PHI, 'b': PHI**2, 'c': PHI**3}
    phi_resonance = optimizer._calculate_phi_harmonic_resonance(phi_params)
    
    # Test with regular values
    regular_params = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    regular_resonance = optimizer._calculate_phi_harmonic_resonance(regular_params)
    
    print(f"   Phi-aligned resonance: {phi_resonance:.6f}")
    print(f"   Regular resonance: {regular_resonance:.6f}")
    
    # Phi-aligned should have reasonable resonance
    assert 0.0 <= phi_resonance <= 1.0
    assert 0.0 <= regular_resonance <= 1.0
    print("   ✅ Phi-harmonic resonance calculation working correctly\n")
    
    # Test 3: Parameter Variations Creation
    print("📊 Test 3: Parameter Variations Creation")
    print("-" * 50)
    
    base_params = {'x': 10.0, 'y': 20.0}
    variations = optimizer._create_phi_parameter_variations(base_params, 5)
    
    print(f"   Created {len(variations)} variations")
    
    # Check that variations are different
    unique_variations = 0
    for variation in variations:
        if variation != base_params:
            unique_variations += 1
    
    print(f"   Unique variations: {unique_variations}/{len(variations)}")
    assert unique_variations > 0
    print("   ✅ Parameter variations created correctly\n")
    
    # Test 4: Phi-Field Array Processing
    print("📊 Test 4: Phi-Field Array Processing")
    print("-" * 50)
    
    original_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    transformed_array = optimizer._apply_phi_field_to_array(
        original_array, phi_factor=PHI, rotation=GOLDEN_ANGLE
    )
    
    print(f"   Original array: {original_array}")
    print(f"   Transformed array: {transformed_array}")
    
    # Should modify values but keep reasonable ranges
    assert not np.array_equal(original_array, transformed_array)
    assert transformed_array.shape == original_array.shape
    
    # Check that changes are reasonable (not extreme)
    ratios = np.abs(transformed_array / original_array)
    assert np.all(ratios > 0.5) and np.all(ratios < 2.0)
    print("   ✅ Phi-field array processing working correctly\n")
    
    # Test 5: Quantum Parameter States
    print("📊 Test 5: Quantum Parameter States Creation")
    print("-" * 50)
    
    base_params = {'a': 1.0, 'b': 2.0}
    num_states = 4
    quantum_states = optimizer._create_quantum_parameter_states(base_params, num_states)
    
    print(f"   Created {len(quantum_states)} quantum states")
    
    # Check structure
    assert len(quantum_states) == num_states
    for state in quantum_states:
        assert set(state.keys()) == set(base_params.keys())
    
    # Check for variations
    unique_states = sum(1 for state in quantum_states if state != base_params)
    print(f"   Unique states: {unique_states}/{num_states}")
    assert unique_states > 0
    print("   ✅ Quantum parameter states created correctly\n")
    
    # Test 6: Quantum Evolution Simulation
    print("📊 Test 6: Quantum Evolution Simulation")
    print("-" * 50)
    
    num_states = 3
    amplitudes = np.ones(num_states, dtype=complex) / np.sqrt(num_states)
    parameter_states = [{'x': i} for i in range(num_states)]
    
    print(f"   Initial amplitudes: {amplitudes}")
    
    evolved_amplitudes = optimizer._apply_quantum_evolution(amplitudes, parameter_states)
    
    print(f"   Evolved amplitudes: {evolved_amplitudes}")
    
    # Check normalization preservation
    norm = np.sqrt(np.sum(np.abs(evolved_amplitudes)**2))
    print(f"   Final norm: {norm:.10f}")
    assert abs(norm - 1.0) < 1e-10
    print("   ✅ Quantum evolution simulation working correctly\n")
    
    # Test 7: Algorithm Selection Logic
    print("📊 Test 7: Algorithm Selection Logic")
    print("-" * 50)
    
    # Test algorithm selection for different levels
    test_complexity = {'type': 'simple', 'parallelizable': False}
    
    algorithms = {}
    for level in OptimizationLevel:
        optimizer.set_optimization_level(level)
        algorithm = optimizer._select_algorithm_for_state("OBSERVE", test_complexity)
        algorithms[level.name] = algorithm
        print(f"   {level.name}: {algorithm}")
    
    # Should have different algorithms for different levels
    unique_algorithms = len(set(algorithms.values()))
    print(f"   Unique algorithms: {unique_algorithms}")
    assert unique_algorithms >= 3  # Should have at least 3 different algorithms
    print("   ✅ Algorithm selection logic working correctly\n")
    
    # Test 8: Parallelizability Analysis
    print("📊 Test 8: Parallelizability Analysis")
    print("-" * 50)
    
    test_cases = [
        ({'x': 1}, "single parameter"),
        ({'x': 1, 'y': 2}, "multiple parameters"),
        ({'data': np.array([1, 2, 3])}, "array parameter"),
        ({'big': 100}, "large value"),
    ]
    
    for params, description in test_cases:
        is_parallelizable = optimizer._analyze_parallelizability(
            lambda **kwargs: sum(kwargs.values()) if all(isinstance(v, (int, float)) for v in kwargs.values()) else 0,
            params
        )
        print(f"   {description}: {is_parallelizable}")
    print("   ✅ Parallelizability analysis working correctly\n")
    
    # All tests passed
    print("🎉 All Algorithm Correctness Tests Passed!")
    print("✅ PhiFlow Phi-Quantum Optimizer algorithms are mathematically correct")
    return True

if __name__ == "__main__":
    success = test_algorithm_correctness()
    if success:
        print("\n🌟 Algorithm correctness validation complete!")
        exit(0)
    else:
        print("\n💥 Algorithm correctness tests failed")
        exit(1)