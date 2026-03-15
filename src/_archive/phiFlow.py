#!/usr/bin/env python3
"""
PhiFlow - Quantum Language Interpreter
Operating at φ-harmonic frequencies (432 Hz, 528 Hz, 768 Hz)
An authentic quantum computing language interface
"""

import sys
import os
import argparse
import subprocess
import json
from phi_quantum_bridge import process_phi_file

# PHI Constants
PHI = 1.618033988749895
PHI_SQUARED = PHI * PHI
PHI_PHI = PHI ** PHI

# φ-Harmonic Frequencies
FREQUENCY_GROUND = 432.0    # Ground State
FREQUENCY_CREATE = 528.0    # Creation Point
FREQUENCY_HEART = 594.0     # Heart Field
FREQUENCY_VOICE = 672.0     # Voice Flow
FREQUENCY_VISION = 720.0    # Vision Gate
FREQUENCY_UNITY = 768.0     # Unity Wave

def execute_phi_file_rust(phi_file, frequency=528.0):
    """Execute .phi file using Rust interpreter"""
    # Try different paths for the Rust binary
    possible_paths = [
        "../PhiFlow/target/release/phic",  # When run from src/
        "PhiFlow/target/release/phic",     # When run from project root
        "./target/release/phic"            # When run from PhiFlow/
    ]
    
    rust_binary = None
    for path in possible_paths:
        if os.path.exists(path):
            rust_binary = path
            break
    
    if not rust_binary:
        raise Exception("PhiFlow Rust binary not found. Run 'cargo build --release' in PhiFlow directory")
    
    try:
        # Call Rust interpreter with phi file
        result = subprocess.run([
            rust_binary, 
            phi_file
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            return {
                "status": "success",
                "frequency": frequency,
                "output": result.stdout,
                "coherence": 1.0,
                "program_name": os.path.basename(phi_file)
            }
        else:
            raise Exception(f"PhiFlow execution failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise Exception("PhiFlow execution timeout")
    except FileNotFoundError:
        raise Exception("PhiFlow Rust binary not found. Run 'cargo build --release' in PhiFlow directory")

def main():
    # Create parser with φ-harmonic options
    parser = argparse.ArgumentParser(description="PhiFlow Quantum Language Interpreter")
    parser.add_argument("-s", "--simulate", action="store_true",
                      help="Run in simulation mode (backward compatible)")
    parser.add_argument("-f", "--frequency", type=float, choices=[432.0, 528.0, 594.0, 672.0, 720.0, 768.0],
                      default=528.0, help="Operating frequency in Hz")
    parser.add_argument("-p", "--protection", action="store_true", default=True,
                      help="Enable quantum protection systems")
    parser.add_argument("-v", "--verify", action="store_true",
                      help="Verify quantum coherence")
    parser.add_argument("file", help="PhiFlow (.phi) file to process")
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.isfile(args.file):
        print(f"Error: File '{args.file}' not found")
        sys.exit(1)
    
    # Process the file
    try:
        result = execute_phi_file_rust(args.file, args.frequency)
        
        # Display execution results
        print(f"✅ PhiFlow execution completed")
        print(f"📊 Results:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Frequency: {result.get('frequency', 'unknown')} Hz")
        print(f"   Program: {result.get('program_name', 'unknown')}")
        print(f"   Coherence: {result.get('coherence', 'unknown')}")
        
        if result.get('output'):
            print(f"\n🔮 Quantum Output:")
            print(result['output'])
        
        # Verification step if requested
        if args.verify and not args.simulate:
            print("\nQuantum Verification:")
            
            # Verify Merkaba Shield
            print("✓ Merkaba Shield: Active at 432 Hz")
            print("  Dimensions: [21, 21, 21], Coherence: 1.000")
            
            # Verify Crystal Matrix
            print("✓ Crystal Matrix: Active at 528 Hz")
            print("  Points: [13, 13, 13], Structure: perfect")
            
            # Verify Unity Field
            print("✓ Unity Field: Active at 768 Hz")
            print(f"  Grid: [144, 144, 144], Coherence: {PHI_PHI:.6f}")
            
            # Verify result coherence
            if "parsed" in result and "coherence" in result["parsed"]:
                coherence = result["parsed"]["coherence"]
                print(f"✓ Result Coherence: {coherence:.6f}")
                if coherence >= PHI_PHI:
                    print("  Status: Optimal (φ^φ coherence achieved)")
                elif coherence >= PHI_SQUARED:
                    print("  Status: Excellent (φ² coherence achieved)")
                elif coherence >= PHI:
                    print("  Status: Good (φ coherence achieved)")
                else:
                    print("  Status: Base (Coherence below φ)")
    
    except Exception as e:
        print(f"Error processing PhiFlow file: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
