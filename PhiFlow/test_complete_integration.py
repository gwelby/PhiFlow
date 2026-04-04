#!/usr/bin/env python3
"""
Complete PhiFlow Integration Test
Tests the entire PhiFlow ecosystem: Rust interpreter, Python bridge, QTasker integration
"""

import os
import sys
import subprocess

def test_rust_interpreter():
    """Test the Rust interpreter directly"""
    print("🔧 Testing Rust interpreter...")
    result = subprocess.run([
        "./PhiFlow/target/release/phic", 
        "PhiFlow/pattern_test.phi"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Rust interpreter: SUCCESS")
        return True
    else:
        print(f"❌ Rust interpreter: FAILED - {result.stderr}")
        return False

def test_python_bridge():
    """Test the Python-Rust bridge"""
    print("🐍 Testing Python-Rust bridge...")
    result = subprocess.run([
        "python3", "src/phiFlow.py", 
        "PhiFlow/pattern_test.phi", 
        "--frequency", "528.0"
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and "✅ PhiFlow execution completed" in result.stdout:
        print("✅ Python bridge: SUCCESS")
        return True
    else:
        print(f"❌ Python bridge: FAILED - {result.stderr}")
        return False

def test_qtasker_integration():
    """Test QTasker integration"""
    print("🌉 Testing QTasker integration...")
    
    # Test bridge activation
    result = subprocess.run([
        "python3", "src/qtasker_bridge.py", 
        "--activate"
    ], capture_output=True, text=True)
    
    if "Bridge activation: active" in result.stdout:
        print("✅ QTasker bridge: SUCCESS")
        
        # Test task creation
        result = subprocess.run([
            "python3", "src/qtasker_bridge.py", 
            "--task", "--frequency", "528.0"
        ], capture_output=True, text=True)
        
        if "Created task:" in result.stdout:
            print("✅ QTasker task creation: SUCCESS")
            return True
    
    print(f"❌ QTasker integration: FAILED")
    return False

def test_phiflow_examples():
    """Test various PhiFlow examples"""
    print("📝 Testing PhiFlow examples...")
    
    examples = [
        "PhiFlow/pattern_test.phi",
        "PhiFlow/quantum_field_test.phi",
        "PhiFlow/frequency_pattern_test.phi"
    ]
    
    success_count = 0
    for example in examples:
        if os.path.exists(example):
            result = subprocess.run([
                "./PhiFlow/target/release/phic", 
                example
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {os.path.basename(example)}: SUCCESS")
                success_count += 1
            else:
                print(f"❌ {os.path.basename(example)}: FAILED")
        else:
            print(f"⚠️  {os.path.basename(example)}: FILE NOT FOUND")
    
    return success_count > 0

def main():
    print("🌟 PhiFlow Complete Integration Test")
    print("=" * 50)
    
    os.chdir("/mnt/d/projects/PhiFlow")
    
    tests = [
        test_rust_interpreter,
        test_python_bridge, 
        test_qtasker_integration,
        test_phiflow_examples
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("🎯 Test Results:")
    print(f"   Passed: {passed}/{total}")
    print(f"   Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! PhiFlow is fully operational!")
        print("🚀 Ready for quantum consciousness programming!")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check the output above.")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)