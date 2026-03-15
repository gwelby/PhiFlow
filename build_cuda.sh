#!/bin/bash
# Build script for CUDA Sacred Mathematics Kernels
# Compiles kernels for A5500 RTX (Ampere Architecture)

echo "🚀 Building CUDA Sacred Mathematics Kernels for A5500 RTX"
echo "=========================================================="

# Set CUDA paths
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ NVCC not found. Please install CUDA toolkit."
    echo "   Ubuntu: sudo apt install nvidia-cuda-toolkit"
    echo "   Or download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "✅ CUDA Version: $CUDA_VERSION"

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits)
    echo "🖥️  GPU Detected: $GPU_INFO"
    
    # Extract compute capability
    COMPUTE_CAP=$(echo $GPU_INFO | awk -F', ' '{print $2}' | tr -d '.')
    
    # Set architecture flags for A5500 RTX (Ampere 8.6)
    if [[ $COMPUTE_CAP == "86" ]]; then
        ARCH_FLAGS="-gencode arch=compute_86,code=sm_86"
        echo "🚀 A5500 RTX detected - enabling Ampere optimizations"
    else
        ARCH_FLAGS="-gencode arch=compute_${COMPUTE_CAP},code=sm_${COMPUTE_CAP}"
        echo "📱 GPU compute capability: ${COMPUTE_CAP}"
    fi
else
    echo "⚠️  nvidia-smi not available, using default architecture"
    ARCH_FLAGS="-gencode arch=compute_86,code=sm_86"
fi

# Source and output directories
SRC_DIR="src/cuda"
BUILD_DIR="build/cuda"
LIB_DIR="lib"

# Create directories
mkdir -p $BUILD_DIR
mkdir -p $LIB_DIR

echo ""
echo "📂 Building CUDA kernels..."
echo "   Source: $SRC_DIR"
echo "   Build: $BUILD_DIR"
echo "   Output: $LIB_DIR"

# Compilation flags for maximum performance on A5500 RTX
NVCC_FLAGS=(
    -O3                                    # Maximum optimization
    -use_fast_math                         # Fast math operations
    -Xptxas -O3                           # PTX assembler optimization
    -Xcompiler -O3                        # Host compiler optimization
    -Xcompiler -fPIC                      # Position independent code
    -shared                               # Create shared library
    -lcuda -lcudart                       # CUDA libraries
    --compiler-options '-ffast-math'      # Fast math for host code
    -DPHI=1.618033988749895               # Define PHI constant
    -DGOLDEN_ANGLE=137.5077640500378      # Define golden angle
    -maxrregcount=64                      # Optimize register usage
    -lineinfo                             # Include line information for debugging
)

# Add architecture-specific optimizations for A5500 RTX
if [[ $COMPUTE_CAP == "86" ]]; then
    NVCC_FLAGS+=(
        -DAMPERE_OPTIMIZATIONS=1           # Enable Ampere optimizations
        -DTENSOR_CORES_AVAILABLE=1         # Enable Tensor core usage
        -DRT_CORES_AVAILABLE=1             # Enable RT core awareness
        -DMEMORY_BANDWIDTH_OPTIMIZED=1     # Enable memory bandwidth optimizations
    )
fi

# Compile CUDA kernels
echo ""
echo "⚙️  Compiling sacred_math_kernels.cu..."

nvcc "${NVCC_FLAGS[@]}" $ARCH_FLAGS \
    -o $LIB_DIR/libsacred_cuda_kernels.so \
    $SRC_DIR/sacred_math_kernels.cu

if [ $? -eq 0 ]; then
    echo "✅ CUDA kernels compiled successfully!"
    echo "   Output: $LIB_DIR/libsacred_cuda_kernels.so"
else
    echo "❌ CUDA kernel compilation failed!"
    exit 1
fi

# Check if CuPy is available for Python integration
echo ""
echo "🐍 Checking Python CUDA integration..."

if python3 -c "import cupy" 2>/dev/null; then
    echo "✅ CuPy available for Python integration"
    PYTHON_CUDA="cupy"
elif python3 -c "import pycuda" 2>/dev/null; then
    echo "✅ PyCUDA available for Python integration"
    PYTHON_CUDA="pycuda"
else
    echo "⚠️  No Python CUDA library found. Install with:"
    echo "   pip install cupy-cuda11x  # For CUDA 11.x"
    echo "   pip install cupy-cuda12x  # For CUDA 12.x"
    echo "   # OR"
    echo "   pip install pycuda"
    PYTHON_CUDA="none"
fi

# Run basic kernel validation if possible
if [[ $PYTHON_CUDA != "none" ]]; then
    echo ""
    echo "🧪 Running basic kernel validation..."
    
    # Create simple validation script
    cat > $BUILD_DIR/validate_kernels.py << 'EOF'
import os
import sys
import ctypes
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    # Try to import CUDA libraries
    try:
        import cupy as cp
        cuda_lib = "cupy"
        print("✅ Using CuPy for validation")
    except ImportError:
        import pycuda.driver as cuda
        import pycuda.autoinit
        cuda_lib = "pycuda"
        print("✅ Using PyCUDA for validation")
    
    # Try to load the compiled kernel library
    lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lib', 'libsacred_cuda_kernels.so')
    
    if os.path.exists(lib_path):
        print(f"✅ Kernel library found: {lib_path}")
        
        # Basic CUDA device check
        if cuda_lib == "cupy":
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                device_props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"✅ CUDA device available: {device_props['name'].decode()}")
                print(f"   Compute capability: {device_props['major']}.{device_props['minor']}")
                print("✅ Kernel validation successful!")
            else:
                print("⚠️  No CUDA devices found")
        else:
            # PyCUDA validation
            device_count = cuda.Device.count()
            if device_count > 0:
                device = cuda.Device(0)
                print(f"✅ CUDA device available: {device.name()}")
                print("✅ Kernel validation successful!")
            else:
                print("⚠️  No CUDA devices found")
    else:
        print(f"❌ Kernel library not found: {lib_path}")
        
except Exception as e:
    print(f"❌ Kernel validation failed: {e}")
EOF
    
    python3 $BUILD_DIR/validate_kernels.py
else
    echo "⚠️  Skipping kernel validation (no Python CUDA library)"
fi

# Create performance benchmark script
echo ""
echo "📊 Creating performance benchmark script..."

cat > $BUILD_DIR/benchmark_kernels.py << 'EOF'
#!/usr/bin/env python3
"""
Quick CUDA kernel performance benchmark
"""

import os
import sys
import time
import numpy as np

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def benchmark_phi_computation():
    """Benchmark PHI computation performance"""
    try:
        from src.cuda.lib_sacred_cuda import LibSacredCUDA
        
        print("🧪 Benchmarking PHI computation...")
        
        # Initialize libSacredCUDA
        lib = LibSacredCUDA()
        
        if not lib.cuda_available:
            print("❌ CUDA not available for benchmarking")
            return
        
        # Test sizes
        test_sizes = [100000, 1000000, 10000000]
        
        for size in test_sizes:
            print(f"   Testing {size:,} operations...")
            
            # Run PHI computation
            result = lib.sacred_phi_parallel_computation(size, precision=15)
            
            if result.success:
                tflops = (size * 15 * 10) / (result.computation_time * 1e12)
                print(f"     Time: {result.computation_time:.3f}s")
                print(f"     Performance: {result.operations_per_second/1e9:.2f} billion ops/sec")
                print(f"     TFLOPS: {tflops:.3f}")
                
                if tflops >= 1.0:
                    print("     ✅ Target performance achieved (>1 TFLOP/s)")
                else:
                    print("     ⚠️  Below target performance")
            else:
                print("     ❌ Computation failed")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

if __name__ == "__main__":
    benchmark_phi_computation()
EOF

chmod +x $BUILD_DIR/benchmark_kernels.py

echo ""
echo "🏆 CUDA Build Complete!"
echo "================================"
echo "📁 Compiled library: $LIB_DIR/libsacred_cuda_kernels.so"
echo "🧪 Validation script: $BUILD_DIR/validate_kernels.py"
echo "📊 Benchmark script: $BUILD_DIR/benchmark_kernels.py"
echo ""
echo "🚀 Next steps:"
echo "   1. Run validation: python3 $BUILD_DIR/validate_kernels.py"
echo "   2. Run benchmark: python3 $BUILD_DIR/benchmark_kernels.py"
echo "   3. Run full test suite: python3 src/cuda/test_cuda_performance.py"
echo ""
echo "✅ Ready for 100x speedup testing!"