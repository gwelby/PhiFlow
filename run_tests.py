#!/usr/bin/env python3
"""
Comprehensive test runner for PhiFlow Quantum Consciousness Engine
Provides various test execution modes and reporting options
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description or cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="PhiFlow Test Runner")
    parser.add_argument("--mode", choices=[
        "all", "unit", "integration", "performance", "phase0", "phase1", 
        "phase2", "phase3", "consciousness", "quantum", "cuda", "sacred-math",
        "coverage", "benchmark", "quick", "slow"
    ], default="all", help="Test execution mode")
    
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--markers", help="Custom pytest markers")
    parser.add_argument("--file", help="Run specific test file")
    parser.add_argument("--function", help="Run specific test function")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd_parts = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd_parts.append("-v")
    
    # Add coverage if requested
    if args.coverage or args.mode == "coverage":
        cmd_parts.extend([
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml"
        ])
    
    # Add HTML report if requested
    if args.html:
        cmd_parts.extend(["--html=reports/test_report.html", "--self-contained-html"])
    
    # Add parallel execution if requested
    if args.parallel:
        cmd_parts.extend(["-n", "auto"])
    
    # Add mode-specific markers
    mode_markers = {
        "unit": "unit",
        "integration": "integration",
        "performance": "performance",
        "phase0": "phase0",
        "phase1": "phase1",
        "phase2": "phase2",
        "phase3": "phase3",
        "consciousness": "consciousness",
        "quantum": "quantum",
        "cuda": "cuda",
        "sacred-math": "sacred_math",
        "quick": "not slow",
        "slow": "slow",
        "benchmark": "benchmark"
    }
    
    if args.mode in mode_markers:
        cmd_parts.extend(["-m", mode_markers[args.mode]])
    
    # Add custom markers
    if args.markers:
        cmd_parts.extend(["-m", args.markers])
    
    # Add specific file or function
    if args.file:
        if args.function:
            cmd_parts.append(f"{args.file}::{args.function}")
        else:
            cmd_parts.append(args.file)
    elif args.function:
        cmd_parts.append(f"-k {args.function}")
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    # Run the tests
    cmd = " ".join(cmd_parts)
    success = run_command(cmd, f"PhiFlow Tests - Mode: {args.mode}")
    
    # Additional reporting based on mode
    if args.mode == "coverage":
        print("\n" + "="*60)
        print("COVERAGE SUMMARY")
        print("="*60)
        run_command("python -m coverage report", "Coverage Summary")
        
        print("\nHTML coverage report generated at: htmlcov/index.html")
    
    elif args.mode == "performance":
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUMMARY")
        print("="*60)
        print("Performance tests completed. Check output above for benchmark results.")
    
    elif args.mode == "benchmark":
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        print("Benchmark tests completed. Results saved to reports/benchmark_results.json")
    
    # Print summary
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Success: {'✓' if success else '✗'}")
    
    if args.html:
        print(f"HTML Report: reports/test_report.html")
    
    if args.coverage or args.mode == "coverage":
        print(f"Coverage Report: htmlcov/index.html")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())