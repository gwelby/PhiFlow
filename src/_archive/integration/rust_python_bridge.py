#!/usr/bin/env python3
"""
PhiFlow Rust-Python FFI Bridge
Connects Rust quantum and consciousness components to Python execution environment
"""

import ctypes
import ctypes.util
import os
import sys
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import threading
import time

# PHI constants for bridge operations
PHI = 1.618033988749895
GOLDEN_ANGLE = 137.5077640

@dataclass
class QuantumCircuitResult:
    """Result from Rust quantum circuit execution"""
    status: str
    counts: Dict[str, int]
    execution_time: float
    coherence: float
    metadata: Dict[str, Any]
    success: bool

@dataclass
class ConsciousnessMetrics:
    """Consciousness monitoring metrics from Rust"""
    coherence: float
    clarity: float
    flow_state: float
    attention_level: float
    sacred_frequency: Optional[int]
    phi_alignment: float
    timestamp: float

@dataclass
class CUDAProcessingResult:
    """CUDA processing result from Rust"""
    result_data: np.ndarray
    execution_time: float
    tflops_achieved: float
    speedup_ratio: float
    success: bool
    error_message: Optional[str]

class RustPythonBridge:
    """
    FFI Bridge connecting Rust components to Python integration engine
    
    Provides high-level Python interface to:
    - Rust quantum simulation and circuit execution
    - Consciousness monitoring and EEG processing
    - CUDA-accelerated sacred mathematics
    - System coherence monitoring
    """
    
    def __init__(self):
        """Initialize the Rust-Python FFI bridge"""
        self.rust_lib = None
        self.initialized = False
        self.lock = threading.RLock()
        
        # Connection status tracking
        self.quantum_bridge_active = False
        self.consciousness_bridge_active = False
        self.cuda_bridge_active = False
        
        # Performance metrics
        self.bridge_metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'average_latency': 0.0,
            'rust_lib_version': None
        }
        
        print("🌉 PhiFlow Rust-Python Bridge initializing...")
        
    def initialize(self) -> bool:
        """
        Initialize the FFI bridge and load the Rust library
        
        Returns:
            Success status of initialization
        """
        try:
            with self.lock:
                if self.initialized:
                    return True
                
                # Find the Rust library
                lib_path = self._find_rust_library()
                if not lib_path:
                    print("❌ Could not locate PhiFlow Rust library")
                    return False
                
                print(f"📚 Loading Rust library: {lib_path}")
                
                # Load the library
                self.rust_lib = ctypes.CDLL(lib_path)
                
                # Initialize function signatures
                self._setup_function_signatures()
                
                # Initialize the Rust library
                init_result = self.rust_lib.phiflow_initialize()
                if init_result != 0:
                    print(f"❌ Rust library initialization failed: {init_result}")
                    return False
                
                # Get library version
                version_ptr = self.rust_lib.phiflow_get_version()
                if version_ptr:
                    version = ctypes.string_at(version_ptr).decode('utf-8')
                    self.bridge_metrics['rust_lib_version'] = version
                    print(f"✅ Rust library version: {version}")
                
                # Test bridge connections
                self.quantum_bridge_active = self._test_quantum_bridge()
                self.consciousness_bridge_active = self._test_consciousness_bridge()
                self.cuda_bridge_active = self._test_cuda_bridge()
                
                self.initialized = True
                
                print("✅ Rust-Python Bridge initialized successfully!")
                print(f"   🔬 Quantum Bridge: {'✅' if self.quantum_bridge_active else '❌'}")
                print(f"   🧠 Consciousness Bridge: {'✅' if self.consciousness_bridge_active else '❌'}")
                print(f"   ⚡ CUDA Bridge: {'✅' if self.cuda_bridge_active else '❌'}")
                
                return True
                
        except Exception as e:
            print(f"❌ Bridge initialization failed: {e}")
            return False
    
    def execute_quantum_circuit(self, circuit_json: str, backend: Optional[str] = None) -> QuantumCircuitResult:
        """
        Execute a quantum circuit using Rust quantum backend
        
        Args:
            circuit_json: JSON representation of quantum circuit
            backend: Optional backend name (defaults to simulator)
            
        Returns:
            QuantumCircuitResult: Execution results
        """
        if not self.initialized or not self.quantum_bridge_active:
            return QuantumCircuitResult(
                status="BRIDGE_ERROR",
                counts={},
                execution_time=0.0,
                coherence=0.0,
                metadata={"error": "Quantum bridge not available"},
                success=False
            )
        
        try:
            with self.lock:
                start_time = time.time()
                
                # Convert inputs to C-compatible types
                circuit_cstr = ctypes.c_char_p(circuit_json.encode('utf-8'))
                backend_cstr = ctypes.c_char_p(backend.encode('utf-8') if backend else b"simulator")
                
                # Call Rust function
                result_ptr = self.rust_lib.phiflow_execute_quantum_circuit(circuit_cstr, backend_cstr)
                
                if not result_ptr:
                    return QuantumCircuitResult(
                        status="EXECUTION_FAILED",
                        counts={},
                        execution_time=time.time() - start_time,
                        coherence=0.0,
                        metadata={"error": "Rust execution returned null"},
                        success=False
                    )
                
                # Parse result JSON from Rust
                result_json = ctypes.string_at(result_ptr).decode('utf-8')
                result_data = json.loads(result_json)
                
                # Free Rust memory
                self.rust_lib.phiflow_free_string(result_ptr)
                
                # Update metrics
                execution_time = time.time() - start_time
                self._update_bridge_metrics(execution_time, True)
                
                return QuantumCircuitResult(
                    status=result_data.get('status', 'UNKNOWN'),
                    counts=result_data.get('counts', {}),
                    execution_time=execution_time,
                    coherence=result_data.get('coherence', 0.0),
                    metadata=result_data.get('metadata', {}),
                    success=result_data.get('success', False)
                )
                
        except Exception as e:
            self._update_bridge_metrics(time.time() - start_time, False)
            print(f"❌ Quantum circuit execution failed: {e}")
            return QuantumCircuitResult(
                status="BRIDGE_ERROR",
                counts={},
                execution_time=0.0,
                coherence=0.0,
                metadata={"error": str(e)},
                success=False
            )
    
    def get_consciousness_metrics(self) -> ConsciousnessMetrics:
        """
        Get current consciousness monitoring metrics from Rust
        
        Returns:
            ConsciousnessMetrics: Current consciousness state
        """
        if not self.initialized or not self.consciousness_bridge_active:
            return ConsciousnessMetrics(
                coherence=0.0,
                clarity=0.0,
                flow_state=0.0,
                attention_level=0.0,
                sacred_frequency=None,
                phi_alignment=0.0,
                timestamp=time.time()
            )
        
        try:
            with self.lock:
                start_time = time.time()
                
                # Call Rust function
                result_ptr = self.rust_lib.phiflow_get_consciousness_metrics()
                
                if not result_ptr:
                    return ConsciousnessMetrics(
                        coherence=0.0,
                        clarity=0.0,
                        flow_state=0.0,
                        attention_level=0.0,
                        sacred_frequency=None,
                        phi_alignment=0.0,
                        timestamp=time.time()
                    )
                
                # Parse result JSON from Rust
                result_json = ctypes.string_at(result_ptr).decode('utf-8')
                result_data = json.loads(result_json)
                
                # Free Rust memory
                self.rust_lib.phiflow_free_string(result_ptr)
                
                # Update metrics
                self._update_bridge_metrics(time.time() - start_time, True)
                
                return ConsciousnessMetrics(
                    coherence=result_data.get('coherence', 0.0),
                    clarity=result_data.get('clarity', 0.0),
                    flow_state=result_data.get('flow_state', 0.0),
                    attention_level=result_data.get('attention_level', 0.0),
                    sacred_frequency=result_data.get('sacred_frequency'),
                    phi_alignment=result_data.get('phi_alignment', 0.0),
                    timestamp=result_data.get('timestamp', time.time())
                )
                
        except Exception as e:
            self._update_bridge_metrics(0.0, False)
            print(f"❌ Consciousness metrics retrieval failed: {e}")
            return ConsciousnessMetrics(
                coherence=0.0,
                clarity=0.0,
                flow_state=0.0,
                attention_level=0.0,
                sacred_frequency=None,
                phi_alignment=0.0,
                timestamp=time.time()
            )
    
    def execute_cuda_computation(self, computation_type: str, data: np.ndarray, 
                               parameters: Dict[str, Any]) -> CUDAProcessingResult:
        """
        Execute CUDA-accelerated computation using Rust CUDA bridge
        
        Args:
            computation_type: Type of CUDA computation to perform
            data: Input data for computation
            parameters: Computation parameters
            
        Returns:
            CUDAProcessingResult: CUDA execution results
        """
        if not self.initialized or not self.cuda_bridge_active:
            return CUDAProcessingResult(
                result_data=np.array([]),
                execution_time=0.0,
                tflops_achieved=0.0,
                speedup_ratio=0.0,
                success=False,
                error_message="CUDA bridge not available"
            )
        
        try:
            with self.lock:
                start_time = time.time()
                
                # Prepare input data
                computation_cstr = ctypes.c_char_p(computation_type.encode('utf-8'))
                parameters_json = json.dumps(parameters)
                parameters_cstr = ctypes.c_char_p(parameters_json.encode('utf-8'))
                
                # Convert numpy array to C-compatible format
                data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                data_size = ctypes.c_size_t(data.size)
                
                # Call Rust CUDA function
                result_ptr = self.rust_lib.phiflow_execute_cuda_computation(
                    computation_cstr, data_ptr, data_size, parameters_cstr
                )
                
                if not result_ptr:
                    return CUDAProcessingResult(
                        result_data=np.array([]),
                        execution_time=time.time() - start_time,
                        tflops_achieved=0.0,
                        speedup_ratio=0.0,
                        success=False,
                        error_message="Rust CUDA execution returned null"
                    )
                
                # Parse result JSON from Rust
                result_json = ctypes.string_at(result_ptr).decode('utf-8')
                result_data = json.loads(result_json)
                
                # Extract result data
                if 'result_data' in result_data and result_data['result_data']:
                    result_array = np.array(result_data['result_data'], dtype=np.float32)
                else:
                    result_array = np.array([])
                
                # Free Rust memory
                self.rust_lib.phiflow_free_string(result_ptr)
                
                # Update metrics
                execution_time = time.time() - start_time
                self._update_bridge_metrics(execution_time, result_data.get('success', False))
                
                return CUDAProcessingResult(
                    result_data=result_array,
                    execution_time=execution_time,
                    tflops_achieved=result_data.get('tflops_achieved', 0.0),
                    speedup_ratio=result_data.get('speedup_ratio', 0.0),
                    success=result_data.get('success', False),
                    error_message=result_data.get('error_message')
                )
                
        except Exception as e:
            self._update_bridge_metrics(0.0, False)
            print(f"❌ CUDA computation failed: {e}")
            return CUDAProcessingResult(
                result_data=np.array([]),
                execution_time=0.0,
                tflops_achieved=0.0,
                speedup_ratio=0.0,
                success=False,
                error_message=str(e)
            )
    
    def get_system_coherence(self) -> float:
        """
        Get overall system coherence from Rust coherence monitoring
        
        Returns:
            System coherence value (0.0 to 1.0)
        """
        if not self.initialized:
            return 0.0
        
        try:
            with self.lock:
                coherence = self.rust_lib.phiflow_get_system_coherence()
                return max(0.0, min(1.0, coherence))
        except Exception as e:
            print(f"❌ System coherence retrieval failed: {e}")
            return 0.0
    
    def get_bridge_status(self) -> Dict[str, Any]:
        """
        Get comprehensive bridge status and metrics
        
        Returns:
            Bridge status information
        """
        return {
            'initialized': self.initialized,
            'bridges': {
                'quantum': self.quantum_bridge_active,
                'consciousness': self.consciousness_bridge_active,
                'cuda': self.cuda_bridge_active
            },
            'metrics': self.bridge_metrics.copy(),
            'rust_lib_loaded': self.rust_lib is not None,
            'system_coherence': self.get_system_coherence() if self.initialized else 0.0
        }
    
    def shutdown(self):
        """Shutdown the bridge and cleanup resources"""
        try:
            with self.lock:
                if self.initialized and self.rust_lib:
                    # Call Rust cleanup function
                    self.rust_lib.phiflow_shutdown()
                    print("✅ Rust library shutdown complete")
                
                self.initialized = False
                self.quantum_bridge_active = False
                self.consciousness_bridge_active = False
                self.cuda_bridge_active = False
                self.rust_lib = None
                
        except Exception as e:
            print(f"⚠️ Bridge shutdown warning: {e}")
    
    # Private helper methods
    
    def _find_rust_library(self) -> Optional[str]:
        """Find the PhiFlow Rust library"""
        # Search paths for the library
        search_paths = [
            # Debug build
            Path(__file__).parent.parent.parent / "PhiFlow" / "target" / "debug",
            # Release build
            Path(__file__).parent.parent.parent / "PhiFlow" / "target" / "release",
            # System library paths
            Path("/usr/local/lib"),
            Path("/usr/lib"),
        ]
        
        # Library names to try
        lib_names = [
            "libphiflow.so",      # Linux
            "libphiflow.dylib",   # macOS
            "phiflow.dll",        # Windows
            "libphiflow.a",       # Static library fallback
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for lib_name in lib_names:
                    lib_path = search_path / lib_name
                    if lib_path.exists():
                        return str(lib_path)
        
        # Try system library finder as fallback
        system_lib = ctypes.util.find_library("phiflow")
        if system_lib:
            return system_lib
        
        return None
    
    def _setup_function_signatures(self):
        """Setup C function signatures for the Rust library"""
        if not self.rust_lib:
            return
        
        # Core initialization functions
        self.rust_lib.phiflow_initialize.argtypes = []
        self.rust_lib.phiflow_initialize.restype = ctypes.c_int
        
        self.rust_lib.phiflow_shutdown.argtypes = []
        self.rust_lib.phiflow_shutdown.restype = None
        
        self.rust_lib.phiflow_get_version.argtypes = []
        self.rust_lib.phiflow_get_version.restype = ctypes.c_char_p
        
        # Quantum functions
        self.rust_lib.phiflow_execute_quantum_circuit.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.rust_lib.phiflow_execute_quantum_circuit.restype = ctypes.c_char_p
        
        # Consciousness functions
        self.rust_lib.phiflow_get_consciousness_metrics.argtypes = []
        self.rust_lib.phiflow_get_consciousness_metrics.restype = ctypes.c_char_p
        
        # CUDA functions
        self.rust_lib.phiflow_execute_cuda_computation.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_char_p
        ]
        self.rust_lib.phiflow_execute_cuda_computation.restype = ctypes.c_char_p
        
        # System coherence functions
        self.rust_lib.phiflow_get_system_coherence.argtypes = []
        self.rust_lib.phiflow_get_system_coherence.restype = ctypes.c_float
        
        # Memory management
        self.rust_lib.phiflow_free_string.argtypes = [ctypes.c_char_p]
        self.rust_lib.phiflow_free_string.restype = None
    
    def _test_quantum_bridge(self) -> bool:
        """Test quantum bridge functionality"""
        try:
            # Test with a simple quantum circuit
            test_circuit = {
                "qubits": 1,
                "gates": [{"type": "H", "qubit": 0}],
                "measurements": [0]
            }
            
            result = self.execute_quantum_circuit(json.dumps(test_circuit), "simulator")
            return result.success
        except Exception:
            return False
    
    def _test_consciousness_bridge(self) -> bool:
        """Test consciousness bridge functionality"""
        try:
            metrics = self.get_consciousness_metrics()
            return metrics.timestamp > 0
        except Exception:
            return False
    
    def _test_cuda_bridge(self) -> bool:
        """Test CUDA bridge functionality"""
        try:
            # Test with simple array operation
            test_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            test_params = {"operation": "test", "multiplier": 2.0}
            
            result = self.execute_cuda_computation("test_operation", test_data, test_params)
            return result.success
        except Exception:
            return False
    
    def _update_bridge_metrics(self, execution_time: float, success: bool):
        """Update bridge performance metrics"""
        self.bridge_metrics['total_calls'] += 1
        if success:
            self.bridge_metrics['successful_calls'] += 1
        
        # Update average latency with exponential moving average
        if self.bridge_metrics['average_latency'] == 0.0:
            self.bridge_metrics['average_latency'] = execution_time
        else:
            alpha = 0.1  # Smoothing factor
            self.bridge_metrics['average_latency'] = (
                alpha * execution_time + (1 - alpha) * self.bridge_metrics['average_latency']
            )

# Global bridge instance for singleton access
_bridge_instance = None
_bridge_lock = threading.Lock()

def get_rust_python_bridge() -> RustPythonBridge:
    """
    Get the global Rust-Python bridge instance
    
    Returns:
        RustPythonBridge: Global bridge instance
    """
    global _bridge_instance
    
    with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = RustPythonBridge()
        return _bridge_instance

def initialize_bridge() -> bool:
    """
    Initialize the global Rust-Python bridge
    
    Returns:
        Success status of initialization
    """
    bridge = get_rust_python_bridge()
    return bridge.initialize()

def shutdown_bridge():
    """Shutdown the global Rust-Python bridge"""
    global _bridge_instance
    
    with _bridge_lock:
        if _bridge_instance:
            _bridge_instance.shutdown()
            _bridge_instance = None

# Context manager for bridge lifecycle
class RustPythonBridgeContext:
    """Context manager for Rust-Python bridge lifecycle"""
    
    def __init__(self):
        self.bridge = None
    
    def __enter__(self) -> RustPythonBridge:
        self.bridge = get_rust_python_bridge()
        if not self.bridge.initialize():
            raise RuntimeError("Failed to initialize Rust-Python bridge")
        return self.bridge
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.bridge:
            # Don't shutdown global bridge, just clear local reference
            self.bridge = None

# Example usage and testing
if __name__ == "__main__":
    print("🌉 PhiFlow Rust-Python Bridge - Integration Test")
    print("=" * 60)
    
    # Test bridge initialization
    with RustPythonBridgeContext() as bridge:
        print("✅ Bridge context established!")
        
        # Test bridge status
        status = bridge.get_bridge_status()
        print(f"📊 Bridge Status:")
        print(f"   Initialized: {status['initialized']}")
        print(f"   Quantum Bridge: {status['bridges']['quantum']}")
        print(f"   Consciousness Bridge: {status['bridges']['consciousness']}")
        print(f"   CUDA Bridge: {status['bridges']['cuda']}")
        print(f"   System Coherence: {status['system_coherence']:.3f}")
        
        if status['bridges']['quantum']:
            # Test quantum circuit execution
            test_circuit = {
                "qubits": 2,
                "gates": [
                    {"type": "H", "qubit": 0},
                    {"type": "CNOT", "control": 0, "target": 1}
                ],
                "measurements": [0, 1]
            }
            
            result = bridge.execute_quantum_circuit(json.dumps(test_circuit))
            print(f"🔬 Quantum Test: {result.status} - {result.success}")
        
        if status['bridges']['consciousness']:
            # Test consciousness metrics
            metrics = bridge.get_consciousness_metrics()
            print(f"🧠 Consciousness Coherence: {metrics.coherence:.3f}")
            print(f"🌊 Flow State: {metrics.flow_state:.3f}")
        
        print("✅ All bridge tests completed!")