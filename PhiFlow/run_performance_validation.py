#!/usr/bin/env python3
"""
PhiFlow Performance Validation Runner
====================================

Main orchestration script for comprehensive PhiFlow performance validation.

Runs complete validation suite with:
- 100x Speedup Claims Validation
- Sacred Mathematics Performance Analysis
- Consciousness Integration Performance Report
- System-Wide Performance Assessment
- Statistical Analysis with 95% Confidence
- Hardware-Specific Validation (A5500 RTX)
- Performance Monitoring Dashboard
"""

import sys
import os
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
import traceback
import threading

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'PhiFlow' / 'src'))

# Import validation components
from performance_validation_system import (
    PerformanceValidationSystem, 
    ValidationLevel
)
from performance_benchmarking_suite import (
    PerformanceBenchmarkingSuite
)
from performance_monitoring_dashboard import (
    PerformanceMonitoringDashboard
)

def print_banner():
    """Print validation banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║             🚀 PhiFlow Performance Validation System 🚀                     ║
║                                                                              ║
║  Comprehensive Scientific Validation of All Performance Claims              ║
║                                                                              ║
║  • 100x Speedup Claims (CUDA vs CPU)                                        ║
║  • >1 TFLOP/s Sacred Mathematics Performance                                 ║
║  • <10ms EEG-to-CUDA Pipeline Latency                                       ║
║  • 99.9% Coherence Maintenance                                              ║
║  • 1.8x Consciousness Enhancement                                           ║
║  • Statistical Rigor (95% Confidence Level)                                 ║
║  • Hardware-Specific Validation (NVIDIA A5500 RTX)                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def setup_logging(debug: bool = False, output_dir: str = "/mnt/d/Projects/phiflow/validation_results"):
    """Setup logging configuration"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_path / f"validation_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('ValidationRunner')

def check_system_requirements():
    """Check system requirements for validation"""
    print("🔍 Checking System Requirements...")
    print("-" * 50)
    
    requirements_met = True
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"✅ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro} (Requires 3.8+)")
        requirements_met = False
    
    # Check required packages
    required_packages = [
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'pandas', 
        'psutil', 'tkinter'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Available")
        except ImportError:
            print(f"❌ {package}: Missing (pip install {package})")
            requirements_met = False
    
    # Check CUDA availability
    cuda_available = False
    cuda_info = "Not Available"
    
    try:
        import cupy as cp
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            cuda_info = f"{props['name'].decode()} ({props['totalGlobalMem'] / 1024**3:.1f} GB)"
            cuda_available = True
    except ImportError:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            device_count = cuda.Device.count()
            if device_count > 0:
                device = cuda.Device(0)
                cuda_info = device.name()
                cuda_available = True
        except ImportError:
            pass
    
    if cuda_available:
        print(f"✅ CUDA: {cuda_info}")
    else:
        print(f"⚠️ CUDA: {cuda_info} (GPU acceleration disabled)")
    
    # Check PhiFlow components
    print("\n🧩 Checking PhiFlow Components...")
    print("-" * 40)
    
    components = {
        'CUDA Sacred Library': ('cuda.lib_sacred_cuda', 'LibSacredCUDA'),
        'CUDA Consciousness Processor': ('cuda.cuda_optimizer_integration', 'CUDAConsciousnessProcessor'),
        'Phi-Quantum Optimizer': ('optimization.phi_quantum_optimizer', 'PhiQuantumOptimizer'),
        'Integration Engine': ('integration.phi_flow_integration_engine', 'PhiFlowIntegrationEngine'),
        'Coherence Engine': ('coherence.phi_coherence_engine', 'PhiCoherenceEngine')
    }
    
    available_components = 0
    for name, (module, class_name) in components.items():
        try:
            module_obj = __import__(module, fromlist=[class_name])
            getattr(module_obj, class_name)
            print(f"✅ {name}: Available")
            available_components += 1
        except ImportError:
            print(f"⚠️ {name}: Not Available (using mock implementation)")
    
    print(f"\nComponent Availability: {available_components}/{len(components)} ({available_components/len(components)*100:.1f}%)")
    
    return requirements_met, cuda_available, available_components

def run_validation_suite(args):
    """Run the complete validation suite"""
    logger = setup_logging(args.debug, args.output_dir)
    
    print_banner()
    
    # Check system requirements
    requirements_met, cuda_available, components_available = check_system_requirements()
    
    if not requirements_met:
        print("\n❌ System requirements not met. Please install missing dependencies.")
        return False
    
    print(f"\n🎯 Validation Configuration:")
    print(f"   Validation Level: {args.level}")
    print(f"   Output Directory: {args.output_dir}")
    print(f"   CUDA Available: {cuda_available}")
    print(f"   Components Available: {components_available}/5")
    print(f"   Enable Dashboard: {args.dashboard}")
    print(f"   Enable Benchmarks: {args.benchmarks}")
    print("")
    
    validation_results = {}
    dashboard_thread = None
    
    try:
        # Start monitoring dashboard if requested
        if args.dashboard:
            print("📊 Starting Performance Monitoring Dashboard...")
            dashboard = PerformanceMonitoringDashboard(
                update_interval_ms=1000,
                history_minutes=120,  # 2 hours
                enable_alerts=True,
                output_dir=str(Path(args.output_dir) / "monitoring")
            )
            
            dashboard.start_monitoring()
            
            # Run dashboard in separate thread
            dashboard_thread = threading.Thread(target=dashboard.run_dashboard, daemon=True)
            dashboard_thread.start()
            print("✅ Dashboard started (GUI will open)")
        
        # Run benchmarking suite if requested
        if args.benchmarks:
            print("\n🏁 Running Performance Benchmarking Suite...")
            print("=" * 60)
            
            benchmarker = PerformanceBenchmarkingSuite(
                output_dir=str(Path(args.output_dir) / "benchmarks"),
                debug=args.debug
            )
            
            if args.suite == 'all':
                benchmark_results = benchmarker.run_comprehensive_benchmarks()
            else:
                # Run specific benchmark suite
                suite_methods = {
                    'cpu': benchmarker.run_cpu_baseline_benchmarks,
                    'cuda': benchmarker.run_cuda_acceleration_benchmarks,
                    'sacred': benchmarker.run_sacred_mathematics_benchmarks,
                    'consciousness': benchmarker.run_consciousness_processing_benchmarks,
                    'integration': benchmarker.run_system_integration_benchmarks,
                    'memory': benchmarker.run_memory_bandwidth_benchmarks
                }
                
                method = suite_methods.get(args.suite)
                if method:
                    benchmark_results = {args.suite: method()}
                else:
                    benchmark_results = benchmarker.run_comprehensive_benchmarks()
            
            # Save benchmark results
            benchmarker.save_benchmark_results(benchmark_results)
            benchmark_report = benchmarker.generate_benchmark_report(benchmark_results)
            
            print("\n📋 BENCHMARK RESULTS SUMMARY:")
            print("=" * 50)
            print(benchmark_report)
            
            validation_results['benchmarks'] = benchmark_results
        
        # Run performance validation
        print("\n🔬 Running Scientific Performance Validation...")
        print("=" * 60)
        
        validation_level = ValidationLevel(args.level)
        validator = PerformanceValidationSystem(
            validation_level=validation_level,
            output_dir=str(Path(args.output_dir) / "validation"),
            enable_visualization=not args.no_viz,
            debug=args.debug
        )
        
        # Run comprehensive validation
        start_time = time.time()
        validation_results['validation'] = validator.run_comprehensive_validation()
        validation_time = time.time() - start_time
        
        # Generate reports
        report_path = validator.save_validation_report()
        data_path = validator.save_validation_data()
        
        print(f"\n✅ Validation completed in {validation_time:.1f} seconds")
        print(f"📄 Report saved: {report_path}")
        print(f"💾 Data saved: {data_path}")
        
        # Display validation summary
        validation_report = validator.generate_validation_report()
        print("\n📊 VALIDATION RESULTS SUMMARY:")
        print("=" * 50)
        
        # Extract key metrics from report
        lines = validation_report.split('\n')
        in_summary = False
        for line in lines:
            if "EXECUTIVE SUMMARY" in line:
                in_summary = True
                continue
            elif in_summary and line.strip() == "":
                break
            elif in_summary:
                print(line)
        
        # Display detailed results for key claims
        print("\n🎯 KEY PERFORMANCE CLAIMS VALIDATION:")
        print("=" * 50)
        
        key_claims = [
            ("100x Speedup", "cuda_acceleration", "speedup_ratio", 100.0),
            (">1 TFLOP/s Performance", "cuda_acceleration", "tflops_achieved", 1.0),
            ("<10ms EEG-CUDA Latency", "cuda_acceleration", "latency_ms", 10.0),
            ("99.9% Coherence", "consciousness_processing", "coherence_maintained", 0.999),
            ("1.8x Consciousness Enhancement", "consciousness_processing", "consciousness_enhancement", 1.8)
        ]
        
        for claim_name, benchmark, metric, target in key_claims:
            if benchmark in validation_results['validation']:
                results = validation_results['validation'][benchmark]
                for metric_enum, result in results.items():
                    if metric in metric_enum.value:
                        status = "✅ ACHIEVED" if result.target_achieved else "❌ NOT ACHIEVED"
                        confidence = f"[95% CI: {result.confidence_interval_95[0]:.3f}-{result.confidence_interval_95[1]:.3f}]"
                        print(f"{claim_name}: {result.mean:.3f} (target: {target}) {status} {confidence}")
                        break
        
        # Generate final recommendations
        print("\n💡 PERFORMANCE OPTIMIZATION RECOMMENDATIONS:")
        print("=" * 55)
        
        recommendations = []
        
        # Check overall success rate
        total_metrics = 0
        achieved_metrics = 0
        
        for benchmark_results in validation_results['validation'].values():
            for result in benchmark_results.values():
                total_metrics += 1
                if result.target_achieved:
                    achieved_metrics += 1
        
        success_rate = (achieved_metrics / total_metrics * 100) if total_metrics > 0 else 0
        
        if success_rate >= 80:
            recommendations.append("🎉 Excellent performance! System exceeds most targets")
        elif success_rate >= 60:
            recommendations.append("✅ Good performance with room for optimization")
        else:
            recommendations.append("⚠️ Significant performance improvements needed")
        
        if not cuda_available:
            recommendations.append("🔧 Install CUDA runtime for GPU acceleration benefits")
        
        if components_available < 3:
            recommendations.append("🔧 Initialize missing PhiFlow components for full functionality")
        
        recommendations.append("📈 Review detailed reports for component-specific optimizations")
        recommendations.append("🔄 Run regular validation to detect performance regressions")
        
        for rec in recommendations:
            print(f"• {rec}")
        
        print(f"\n🏆 VALIDATION COMPLETE!")
        print(f"Overall Success Rate: {success_rate:.1f}% ({achieved_metrics}/{total_metrics} targets achieved)")
        print(f"Validation Level: {args.level}")
        print(f"Total Runtime: {validation_time:.1f} seconds")
        
        # Keep dashboard running if requested
        if args.dashboard and dashboard_thread:
            print(f"\n📊 Dashboard is running - close GUI window to exit")
            print(f"   Monitoring data saved to: {Path(args.output_dir) / 'monitoring'}")
            try:
                dashboard_thread.join()
            except KeyboardInterrupt:
                print("\n⚠️ Dashboard interrupted by user")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        if args.debug:
            traceback.print_exc()
        return False
    
    finally:
        # Clean up dashboard if running
        if dashboard_thread:
            try:
                dashboard.stop_monitoring()
            except:
                pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="PhiFlow Performance Validation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation
  python run_performance_validation.py --level quick
  
  # Standard validation with dashboard
  python run_performance_validation.py --level standard --dashboard
  
  # Thorough validation with benchmarks
  python run_performance_validation.py --level thorough --benchmarks
  
  # Research-grade validation (takes hours)
  python run_performance_validation.py --level research --benchmarks --dashboard
  
  # CUDA-only validation
  python run_performance_validation.py --suite cuda --level standard
        """
    )
    
    parser.add_argument('--level', 
                       choices=['quick', 'standard', 'thorough', 'research'], 
                       default='standard',
                       help='Validation intensity level (default: standard)')
    
    parser.add_argument('--suite', 
                       choices=['cpu', 'cuda', 'sacred', 'consciousness', 'integration', 'memory', 'all'],
                       default='all',
                       help='Specific benchmark suite to run (default: all)')
    
    parser.add_argument('--output-dir', 
                       default='/mnt/d/Projects/phiflow/validation_results',
                       help='Output directory for results (default: ./validation_results)')
    
    parser.add_argument('--dashboard', 
                       action='store_true',
                       help='Launch real-time monitoring dashboard')
    
    parser.add_argument('--benchmarks', 
                       action='store_true',
                       help='Run comprehensive benchmarking suite')
    
    parser.add_argument('--no-viz', 
                       action='store_true',
                       help='Disable visualizations (for headless systems)')
    
    parser.add_argument('--debug', 
                       action='store_true',
                       help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Run validation suite
    success = run_validation_suite(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()