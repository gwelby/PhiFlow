#!/usr/bin/env python3
"""
Test existing PhiFlow components to verify functionality
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_quantum_bridge():
    """Test the quantum bridge component"""
    print("🧪 Testing Quantum Bridge...")
    try:
        # Import directly from file path
        sys.path.insert(0, os.path.join('src', 'quantum_bridge'))
        from phi_quantum_interface import PhiQuantumBridge
        
        # Initialize bridge
        bridge = PhiQuantumBridge('simulator')
        
        # Test a simple command
        result = bridge.execute_phiflow_command(
            'INITIALIZE', 432, {'coherence': 1.0}
        )
        
        print(f"✅ Quantum Bridge working!")
        print(f"   Coherence: {result['phi_coherence']:.3f}")
        print(f"   Resonance: {result['phi_resonance']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Quantum Bridge failed: {e}")
        return False

def test_consciousness_interface():
    """Test the consciousness interface component"""
    print("\n🧪 Testing Consciousness Interface...")
    try:
        # Import directly from file path
        sys.path.insert(0, os.path.join('src', 'consciousness'))
        from phi_consciousness_interface import ConsciousnessMonitor, PhiConsciousnessIntegrator
        
        # Initialize components
        monitor = ConsciousnessMonitor(enable_biofeedback=False)
        integrator = PhiConsciousnessIntegrator(monitor)
        
        # Test consciousness measurement
        state = monitor.measure_consciousness_state()
        
        print(f"✅ Consciousness Interface working!")
        print(f"   State: {state.state_name}")
        print(f"   Coherence: {state.heart_coherence:.3f}")
        print(f"   Phi Alignment: {state.phi_alignment:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Consciousness Interface failed: {e}")
        return False

def test_main_engine():
    """Test the main engine (with fixed imports)"""
    print("\n🧪 Testing Main Engine...")
    try:
        # We'll test this after fixing the import issues
        print("⚠️ Main engine needs import fixes - will test after Phase 0 completion")
        return True
        
    except Exception as e:
        print(f"❌ Main Engine failed: {e}")
        return False

if __name__ == "__main__":
    print("🔬 PhiFlow Component Verification Test")
    print("=" * 50)
    
    results = []
    results.append(test_quantum_bridge())
    results.append(test_consciousness_interface())
    results.append(test_main_engine())
    
    print(f"\n📊 Test Results:")
    print(f"   Quantum Bridge: {'✅' if results[0] else '❌'}")
    print(f"   Consciousness Interface: {'✅' if results[1] else '❌'}")
    print(f"   Main Engine: {'⚠️' if results[2] else '❌'}")
    
    if all(results[:2]):  # First two components working
        print(f"\n🎉 Core components verified! Ready for Phase 0 completion.")
    else:
        print(f"\n⚠️ Some components need attention.")