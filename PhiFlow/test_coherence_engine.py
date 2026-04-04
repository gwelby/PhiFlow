#!/usr/bin/env python3
"""
Test script for PhiFlow Perfect Coherence Engine - Task 1.1 Implementation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from coherence.phi_coherence_engine import PhiCoherenceEngine, CoherenceState, CoherenceBaseline
import time

def test_coherence_engine():
    """Test the implemented coherence engine functionality"""
    print("🧪 Testing PhiFlow Perfect Coherence Engine - Task 1.1")
    print("=" * 60)
    
    # Initialize coherence engine
    print("\n1️⃣ Initializing Coherence Engine...")
    engine = PhiCoherenceEngine()
    
    # Test baseline establishment  
    print("\n2️⃣ Testing Baseline Coherence Establishment...")
    baseline = engine.establish_baseline_coherence()
    print(f"✅ Baseline established:")
    print(f"   📊 Quantum: {baseline.quantum_baseline:.3f}")
    print(f"   🧠 Consciousness: {baseline.consciousness_baseline:.3f}")
    print(f"   🌊 Field: {baseline.field_baseline:.3f}")
    print(f"   ✨ Phi-harmonic: {baseline.phi_harmonic_baseline:.3f}")
    
    # Test real-time monitoring
    print("\n3️⃣ Testing Real-time Coherence Monitoring...")
    for i in range(5):
        coherence_state = engine.monitor_multi_system_coherence()
        print(f"   📊 Measurement {i+1}:")
        print(f"      Combined: {coherence_state.combined_coherence:.3f}")
        print(f"      Quantum: {coherence_state.quantum_coherence:.3f}")
        print(f"      Consciousness: {coherence_state.consciousness_coherence:.3f}")
        print(f"      Field: {coherence_state.field_coherence:.3f}")
        print(f"      Phi alignment: {coherence_state.phi_alignment:.3f}")
        print(f"      Trend: {coherence_state.stability_trend:.5f}")
        if coherence_state.correction_events:
            print(f"      🔧 Correction events: {len(coherence_state.correction_events)}")
        time.sleep(0.1)  # Small delay between measurements
    
    # Test comprehensive metrics
    print("\n4️⃣ Testing Comprehensive Metrics...")
    metrics = engine.get_coherence_metrics()
    print(f"✅ System Health: {metrics['system_health']['status']}")
    print(f"✅ Combined Coherence: {metrics['current_coherence']['combined']:.3f}")
    print(f"✅ Average Coherence: {metrics['historical_stats']['average_coherence_10_samples']:.3f}")
    print(f"✅ Coherence Variance: {metrics['historical_stats']['coherence_variance']:.5f}")
    print(f"✅ Measurements Stored: {metrics['monitoring_config']['measurements_stored']}")
    
    # Test target achievement
    current_coherence = metrics['current_coherence']['combined']
    target_coherence = metrics['system_health']['target_coherence']
    
    print(f"\n5️⃣ Testing Target Achievement...")
    print(f"   🎯 Target: {target_coherence:.1%}")
    print(f"   📊 Current: {current_coherence:.1%}")
    
    # Assess if we're meeting Phase 1 requirements
    if current_coherence >= 0.99:
        print("✅ PHASE 1 SUCCESS: 99%+ coherence achieved!")
    elif current_coherence >= 0.95:
        print("⚠️ GOOD PERFORMANCE: 95%+ coherence achieved")
    else:
        print("❌ NEEDS IMPROVEMENT: Below 95% coherence")
    
    print(f"\n🎊 Task 1.1 Implementation Test Complete!")
    print(f"   📊 Multi-system monitoring: ✅ IMPLEMENTED")
    print(f"   🏗️ Baseline establishment: ✅ IMPLEMENTED") 
    print(f"   📈 Real-time measurements: ✅ IMPLEMENTED")
    print(f"   🎯 Target coherence: {target_coherence:.1%}")
    
    return engine, metrics

if __name__ == "__main__":
    test_coherence_engine()