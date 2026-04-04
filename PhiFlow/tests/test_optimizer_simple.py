#!/usr/bin/env python3
"""
Simple test for PhiFlow Phi-Quantum Optimizer real algorithms
Focuses on correctness rather than performance timing
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from optimization.phi_quantum_optimizer import (
    PhiQuantumOptimizer, 
    OptimizationLevel, 
    OptimizationResult
)

def test_phi_optimizer_algorithms():
    """Test that the real phi-optimization algorithms work correctly"""
    
    print("🚀 Testing PhiFlow Phi-Quantum Optimizer Real Algorithms")
    print("=" * 60)
    
    # Initialize optimizer
    optimizer = PhiQuantumOptimizer(enable_cuda=False)
    
    # Test function that actually benefits from optimization
    def cpu_intensive_function(n=1000, multiplier=1.0):
        """CPU-intensive function for testing optimization"""
        time.sleep(0.01)  # 10ms base time to make timing meaningful
        
        # Actual mathematical computation
        result = 0
        for i in range(int(n * multiplier)):
            result += np.sin(i * 0.001) * np.cos(i * 0.001)
        
        return result
    
    # Test parameters
    test_params = {'n': 500, 'multiplier': 1.0}
    
    print("🧪 Testing different optimization levels...")
    
    # Test each optimization level
    levels_to_test = [
        OptimizationLevel.LINEAR,
        OptimizationLevel.PHI_ENHANCED,
        OptimizationLevel.PHI_SQUARED,
        OptimizationLevel.PHI_CUBED,
        OptimizationLevel.CONSCIOUSNESS_QUANTUM
    ]
    
    results = {}
    
    for level in levels_to_test:
        print(f"\n📊 Testing {level.name} (Level {level.value})...")
        
        try:
            # Run optimization
            result = optimizer.optimize_computation(
                cpu_intensive_function,
                test_params,
                level
            )
            
            # Store results
            results[level.name] = result
            
            # Print results
            print(f"   ✅ Success: {result.success}")
            print(f"   ⚡ Speedup: {result.speedup_ratio:.3f}x")
            print(f"   📈 Phi alignment: {result.phi_alignment:.3f}")
            print(f"   🔧 Algorithm: {result.algorithm_used}")
            print(f"   ⏱️ Original time: {result.original_execution_time*1000:.2f}ms")
            print(f"   ⚡ Optimized time: {result.optimized_execution_time*1000:.2f}ms")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[level.name] = None
    
    print("\n" + "="*60)
    
    # Validate results
    successful_results = [r for r in results.values() if r and r.success]
    
    print(f"✅ Successful optimizations: {len(successful_results)}/{len(levels_to_test)}")
    
    if successful_results:
        avg_speedup = sum(r.speedup_ratio for r in successful_results) / len(successful_results)
        max_speedup = max(r.speedup_ratio for r in successful_results)
        avg_phi_alignment = sum(r.phi_alignment for r in successful_results) / len(successful_results)
        
        print(f"📊 Average speedup: {avg_speedup:.3f}x")
        print(f"🏆 Maximum speedup: {max_speedup:.3f}x")
        print(f"🎯 Average phi alignment: {avg_phi_alignment:.3f}")
        
        # Check that algorithms are working
        assert len(successful_results) >= 3, "At least 3 optimization levels should work"
        assert avg_speedup > 0.5, "Average speedup should be reasonable"
        assert max_speedup >= 1.0, "At least one optimization should show improvement"
        
        print("\n🌟 All algorithm tests passed!")
        
    else:
        print("❌ No successful optimizations - algorithms need debugging")
        return False
    
    # Test specific algorithm components
    print("\n" + "="*60)
    print("🔬 Testing Individual Algorithm Components...")
    
    # Test phi-harmonic resonance calculation
    test_resonance_params = {
        'phi_aligned': 1.618,
        'phi_squared': 2.618,
        'regular': 5.0
    }
    
    resonance = optimizer._calculate_phi_harmonic_resonance(test_resonance_params)
    print(f"📊 Phi-harmonic resonance: {resonance:.6f}")
    assert 0.0 <= resonance <= 1.0, "Resonance should be in valid range"
    
    # Test parameter variations
    variations = optimizer._create_phi_parameter_variations(test_params, 3)
    print(f"🔄 Created {len(variations)} parameter variations")
    assert len(variations) == 3, "Should create requested number of variations"
    
    # Test quantum parameter states
    quantum_states = optimizer._create_quantum_parameter_states(test_params, 4)
    print(f"⚛️ Created {len(quantum_states)} quantum parameter states")
    assert len(quantum_states) == 4, "Should create requested number of quantum states"
    
    print("\n✅ All individual component tests passed!")
    
    return True

if __name__ == "__main__":
    success = test_phi_optimizer_algorithms()
    if success:
        print("\n🎉 PhiFlow Phi-Quantum Optimizer algorithms are working correctly!")
        exit(0)
    else:
        print("\n💥 Tests failed - algorithms need fixes")
        exit(1)