# PhiFlow.jl — Sacred Mathematics, Quantum Exploration, and Consciousness Metrics
#
# This is the Julia research layer for PhiFlow. It complements the Rust core:
#
#   Rust  = compiler, runtime, production metrics (the product)
#   Julia = GPU kernels, quantum prototyping, metric research (the lab)
#
# When a formula is proven here, it gets ported to Rust for production.

module PhiFlow

using Printf
using Statistics
using LinearAlgebra
using Random

# ─── Sacred Mathematics ───────────────────────────────────────
include("sacred_math.jl")
using .SacredMath

# ─── Consciousness Metrics (research) ─────────────────────────
include("metrics.jl")
using .Metrics

# ─── GPU Kernels (lazy — only loads if CUDA.jl is available) ──
include("gpu_kernels.jl")
using .GPUKernels

# ─── Quantum Lab (lazy — only loads if Yao.jl is available) ───
include("quantum_lab.jl")
using .QuantumLab

# ─── FFI Bridge to Rust core (lazy — only if libphiflow.so) ───
include("ffi_bridge.jl")
using .FFIBridge

# Re-export everything from submodules
using .SacredMath: PHI, PHI_INV, GOLDEN_ANGLE, TRINITY, FIBONACCI_89,
    SACRED_FREQUENCIES, CONSCIOUSNESS_STATES,
    fibonacci, fibonacci_at, phi_power, golden_spiral, flower_of_life,
    frequency_for_state, state_for_frequency,
    coherence_formula, phi_harmonic_series

using .Metrics: SelfCorrelation, FisherInfo, ConsciousnessProxy,
    Trace, compute_l_self, compute_f_model, compute_c_pf,
    mutual_information, normalized_mi, differentiate

using .GPUKernels: sacred_frequency_synthesis, phi_harmonic_field,
    fibonacci_timing, gpu_sacred_frequency_synthesis,
    gpu_phi_harmonic_field, gpu_fibonacci_timing

using .QuantumLab: ghz_circuit, bell_circuit, sacred_frequency_circuit,
    phi_harmonic_circuit, measure_circuit, ghz_coherence_scaling

using .FFIBridge: compile_phi, run_phi, compile_to_openqasm, lib_available

# Explicit exports — these are what `using PhiFlow` brings into scope
export
    # Sacred math
    PHI, PHI_INV, GOLDEN_ANGLE, TRINITY, FIBONACCI_89,
    SACRED_FREQUENCIES, CONSCIOUSNESS_STATES,
    fibonacci, fibonacci_at, phi_power, golden_spiral, flower_of_life,
    frequency_for_state, state_for_frequency,
    coherence_formula, phi_harmonic_series,
    # Metrics
    SelfCorrelation, FisherInfo, ConsciousnessProxy,
    Trace, compute_l_self, compute_f_model, compute_c_pf,
    mutual_information, normalized_mi, differentiate,
    # GPU
    sacred_frequency_synthesis, phi_harmonic_field,
    fibonacci_timing, gpu_sacred_frequency_synthesis,
    gpu_phi_harmonic_field, gpu_fibonacci_timing,
    # Quantum
    ghz_circuit, bell_circuit, sacred_frequency_circuit,
    phi_harmonic_circuit, measure_circuit, ghz_coherence_scaling,
    # FFI
    compile_phi, run_phi, compile_to_openqasm, lib_available

end # module
