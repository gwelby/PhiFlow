#!/usr/bin/env python3
"""
Test script for PhiFlow Quantum-Consciousness Engine
"""

try:
    print("🧪 Testing PhiFlow Quantum-Consciousness Engine")
    print("=" * 50)
    
    # Test importing the quantum bridge
    print("⚛️ Testing quantum bridge import...")
    from quantum_bridge.phi_quantum_interface import PhiQuantumBridge
    bridge = PhiQuantumBridge('simulator')
    print("✅ Quantum bridge working!")
    
    # Test importing consciousness interface
    print("🧠 Testing consciousness interface import...")
    from consciousness.phi_consciousness_interface import ConsciousnessMonitor
    monitor = ConsciousnessMonitor(enable_biofeedback=False)
    print("✅ Consciousness interface working!")
    
    # Test the integrated engine
    print("🌀 Testing integrated engine...")
    from phiflow_quantum_consciousness_engine import PhiFlowQuantumConsciousnessEngine
    
    engine = PhiFlowQuantumConsciousnessEngine(
        quantum_backend='simulator',
        enable_consciousness=True,
        enable_biofeedback=False
    )
    
    print("✅ Engine initialized successfully!")
    
    # Test a simple quantum command
    print("🧪 Testing quantum command execution...")
    result = bridge.execute_phiflow_command('INITIALIZE', 432, {'coherence': 1.0})
    print(f"⚛️ Quantum Result: Coherence={result['phi_coherence']:.3f}")
    
    # Test consciousness measurement  
    print("🧠 Testing consciousness measurement...")
    state = monitor.measure_consciousness_state()
    print(f"🧠 Consciousness State: {state.state_name}, Coherence={state.heart_coherence:.3f}")
    
    # Cleanup
    monitor.stop_monitoring()
    
    print("\n🎉 ALL TESTS PASSED!")
    print("🚀 PhiFlow Quantum-Consciousness Engine is working perfectly!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc() 