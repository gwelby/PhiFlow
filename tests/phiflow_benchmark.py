from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import time
import toml
import platform
import psutil
import json
from datetime import datetime
from math import gcd, log2
import os

# Load quantum configuration
config = toml.load('quantum_config.toml')

# PhiFlow Quantum Constants
PHI = 1.618034
FREQUENCIES = config['quantum']['frequencies']

class PhiFlowBenchmark:
    def __init__(self):
        self.system_info = self._get_system_info()
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results = {}
        
    def _get_system_info(self):
        """Get detailed system information"""
        return {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "system": platform.system(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else "Unknown"
        }
    
    def test_qubit_capacity(self, start_qubits=4, max_qubits=50):
        """Test maximum viable qubit capacity"""
        print("\n🌟 Testing Maximum Qubit Capacity 🌟")
        
        for num_qubits in range(start_qubits, max_qubits + 1, 2):
            try:
                print(f"\nTesting {num_qubits} qubits...")
                
                # Create test circuit
                qc = QuantumCircuit(num_qubits, num_qubits)
                
                # Initialize with phi-based phases
                for i in range(num_qubits):
                    qc.h(i)
                    phase = np.pi * (FREQUENCIES['ground']/FREQUENCIES['unity']) * (1/PHI)**(i+1)
                    qc.p(phase, i)
                
                # Add entanglement
                for i in range(num_qubits-1):
                    qc.cx(i, i+1)
                
                # Measure
                qc.measure_all()
                
                # Test execution
                simulator = AerSimulator()
                transpiled = transpile(qc, simulator)
                result = simulator.run(transpiled, shots=1024).result()
                
                # Calculate memory usage
                process = psutil.Process(os.getpid())
                memory_gb = process.memory_info().rss / (1024**3)
                
                print(f"Success! Memory usage: {memory_gb:.2f} GB")
                
                self.results[num_qubits] = {
                    "status": "success",
                    "memory_gb": memory_gb,
                    "execution_time": result.time_taken
                }
                
            except Exception as e:
                print(f"Failed at {num_qubits} qubits: {str(e)}")
                self.results[num_qubits] = {
                    "status": "failed",
                    "error": str(e)
                }
                break
        
        # Find maximum successful qubits
        max_qubits = max([q for q, r in self.results.items() if r["status"] == "success"])
        print(f"\n✨ Maximum viable qubits: {max_qubits}")
        return max_qubits
    
    def benchmark_performance(self, num_qubits):
        """Run comprehensive performance benchmark"""
        print(f"\n🌟 Running Performance Benchmark ({num_qubits} qubits) 🌟")
        
        # Test different circuit depths
        depths = [5, 10, 20]
        depth_results = {}
        
        for depth in depths:
            # Create test circuit with specified depth
            qc = QuantumCircuit(num_qubits, num_qubits)
            
            # Initialize with phi-based phases
            for i in range(num_qubits):
                qc.h(i)
                phase = np.pi * (FREQUENCIES['ground']/FREQUENCIES['unity']) * (1/PHI)**(i+1)
                qc.p(phase, i)
            
            # Add layers of operations
            for _ in range(depth):
                # Ground frequency operations
                for i in range(num_qubits-1):
                    qc.cx(i, i+1)
                    qc.p(np.pi * (FREQUENCIES['ground']/FREQUENCIES['unity']) * (1/PHI), i)
                
                # Creation frequency operations
                for i in range(num_qubits):
                    qc.rx(np.pi * (FREQUENCIES['create']/FREQUENCIES['unity']) * (1/PHI), i)
                
                # Unity frequency operations
                for i in range(0, num_qubits-1, 2):
                    qc.cx(i, i+1)
                    qc.p(np.pi * (FREQUENCIES['unity']/FREQUENCIES['unity']) * (1/PHI), i+1)
            
            qc.measure_all()
            
            # Test execution
            simulator = AerSimulator()
            start_time = time.time()
            transpiled = transpile(qc, simulator, optimization_level=3)
            result = simulator.run(transpiled, shots=1024).result()
            execution_time = time.time() - start_time
            
            # Calculate quantum volume
            quantum_volume = 2**num_qubits * depth
            
            depth_results[depth] = {
                "execution_time": execution_time,
                "quantum_volume": quantum_volume,
                "circuit_width": num_qubits,
                "circuit_depth": depth
            }
            
            print(f"\nDepth {depth} results:")
            print(f"Execution time: {execution_time:.2f} seconds")
            print(f"Quantum Volume: {quantum_volume}")
        
        return depth_results
    
    def generate_report(self):
        """Generate detailed scientific report"""
        report = {
            "title": "PhiFlow Quantum Computing Benchmark Report",
            "date": self.timestamp,
            "system_information": self.system_info,
            "quantum_configuration": {
                "frequencies": FREQUENCIES,
                "phi": PHI
            },
            "benchmark_results": self.results
        }
        
        # Save report
        with open('phiflow_benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable summary
        summary = f"""
PhiFlow Quantum Computing Benchmark Report
========================================
Date: {self.timestamp}

System Information:
------------------
Processor: {self.system_info['processor']}
Machine: {self.system_info['machine']}
System: {self.system_info['system']}
RAM: {self.system_info['ram_gb']:.1f} GB
CPU Cores: {self.system_info['cpu_count']}
CPU Frequency: {self.system_info['cpu_freq']} MHz

Quantum Configuration:
--------------------
Ground Frequency: {FREQUENCIES['ground']} Hz
Creation Frequency: {FREQUENCIES['create']} Hz
Unity Frequency: {FREQUENCIES['unity']} Hz
Phi: {PHI}

Benchmark Results:
----------------
Maximum Viable Qubits: {max([q for q, r in self.results.items() if r["status"] == "success"])}
"""
        
        # Save summary
        with open('phiflow_benchmark_summary.txt', 'w') as f:
            f.write(summary)
        
        return summary

if __name__ == "__main__":
    print("🌟 Starting PhiFlow Quantum Benchmark 🌟")
    print("========================================")
    
    # Initialize benchmark
    benchmark = PhiFlowBenchmark()
    
    # Test maximum qubit capacity
    max_qubits = benchmark.test_qubit_capacity()
    
    # Run performance benchmark
    performance_results = benchmark.benchmark_performance(max_qubits)
    
    # Generate and print report
    report = benchmark.generate_report()
    print("\nBenchmark Report:")
    print(report)
    
    print("\n✨ Benchmark Complete ✨")
