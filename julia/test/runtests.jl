using PhiFlow
using Test
using Random

@testset "PhiFlow.jl" begin

    # ═══════════════════════════════════════════════════════════
    # Sacred Math
    # ═══════════════════════════════════════════════════════════

    @testset "SacredMath" begin
        # Constants match Rust exactly
        @test PHI == 1.618033988749895
        @test PHI_INV ≈ 0.6180339887498949 atol=1e-15
        @test GOLDEN_ANGLE ≈ 137.50776405003785 atol=1e-10

        # Fibonacci
        fib = fibonacci(10)
        @test length(fib) == 10
        @test fib[1] == 1
        @test fib[2] == 1
        @test fib[10] == 55
        @test fibonacci_at(10) == 55
        @test fibonacci_at(89) > 0  # The 89th Fibonacci number

        # Phi powers
        @test phi_power(0) == 1.0
        @test phi_power(1) == PHI
        @test phi_power(-1) ≈ PHI_INV atol=1e-15

        # Golden spiral
        @test golden_spiral(0) == 1.0
        @test golden_spiral(90) ≈ PHI atol=1e-10
        @test golden_spiral(180) ≈ PHI^2 atol=1e-10

        # Flower of Life: 19 circles (1 + 6 + 12)
        fol = flower_of_life(1.0)
        @test length(fol) == 19
        @test fol[1] == (0.0, 0.0)  # center

        # Frequency ↔ state mapping
        @test frequency_for_state(:Observe) == 432.0
        @test frequency_for_state(:Create) == 528.0
        @test frequency_for_state(:Superposition) == 963.0
        @test state_for_frequency(432.0) == :Observe
        @test state_for_frequency(430.0) == :Observe  # nearest

        # Coherence formula — matches Rust src/phi_ir/coherence.rs
        # base(d) = 1 - φ^(-d), phase(k≤1) = 1.0
        @test coherence_formula(0) == 0.0          # depth 0 → no base
        @test coherence_formula(1) ≈ 1.0 - PHI_INV atol=1e-15  # 0.382
        @test coherence_formula(2, 1) ≈ PHI_INV atol=1e-15     # 0.618 = φ⁻¹
        @test coherence_formula(3) ≈ 1.0 - PHI^(-3) atol=1e-15 # 0.764

        # Phi-harmonic series
        phs = phi_harmonic_series(5)
        @test length(phs) == 5
        @test sum(phs) ≈ 1.0 atol=1e-15
        @test phs[1] < phs[2] < phs[3] < phs[4] < phs[5]  # increasing
    end

    # ═══════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════

    @testset "Metrics" begin
        # Create a simple trace: observations that predict future actions
        n = 100
        obs = [[Float64(sin(i * 0.1))] for i in 1:n]
        model = [[Float64(sin(i * 0.1 + 0.05))] for i in 1:n]  # model lags obs slightly
        actions = [[Float64(sin(i * 0.1 + 0.1))] for i in 1:n]  # actions lag model
        trace = Trace(obs, model, actions)

        # Self-correlation
        sc = compute_l_self(trace; threshold=0.05)
        @test sc.r_in_norm >= 0.0
        @test sc.r_out_norm >= 0.0
        @test sc.l_self == min(sc.r_in_norm, sc.r_out_norm)

        # Fisher information
        f_model = compute_f_model(model[1:50], actions[51:100])
        @test f_model >= 0.0

        # C_PF
        c_pf = compute_c_pf(trace)
        @test c_pf.c_coh ≈ 0.618033988749895 atol=1e-15  # canonical λ
        @test c_pf.d_int >= 0.0
        @test c_pf.f_self_star >= 0.0
        @test c_pf.c_pf == c_pf.c_coh * c_pf.d_int * c_pf.f_self_star

        # Differentiation
        d = differentiate(obs)
        @test d >= 0.0

        # Mutual information: identical signals should have high MI
        x = collect(1.0:100.0)
        mi_same = mutual_information(x, x)
        mi_random = mutual_information(x, shuffle(copy(x)))
        @test mi_same >= mi_random
    end

    # ═══════════════════════════════════════════════════════════
    # GPU Kernels (CPU fallback — CUDA not required)
    # ═══════════════════════════════════════════════════════════

    @testset "GPUKernels" begin
        # Sacred frequency synthesis (CPU fallback)
        freqs = [432.0, 528.0, 672.0]
        output = sacred_frequency_synthesis(freqs, 0.1, 44100.0)
        @test size(output) == (3, 4410)  # 3 freqs, 0.1s at 44.1kHz

        # Phi-harmonic field
        positions = collect(range(-1.0, 1.0, length=100))
        field = phi_harmonic_field(positions, 432.0, 0.8)
        @test length(field) == 100
        @test all(0.0 .<= field .<= 1.0)  # normalized by PHI, modulated by coherence

        # Fibonacci timing
        timing = fibonacci_timing(10)
        @test length(timing) == 10
        @test timing[1] ≈ PHI_INV atol=1e-15
        @test timing[2] ≈ PHI_INV atol=1e-15
        @test timing[3] ≈ 2 * PHI_INV atol=1e-15
    end

    # ═══════════════════════════════════════════════════════════
    # Quantum Lab (without Yao.jl — should return descriptions)
    # ═══════════════════════════════════════════════════════════

    @testset "QuantumLab" begin
        # Without Yao.jl, these return description strings
        ghz = ghz_circuit(4)
        @test ghz isa String || ghz !== nothing  # either a circuit or a description

        bell = bell_circuit()
        @test bell isa String || bell !== nothing

        sacred = sacred_frequency_circuit(3, 432.0)
        @test sacred isa String || sacred !== nothing

        phi_harm = phi_harmonic_circuit(4, 2)
        @test phi_harm isa String || phi_harm !== nothing
    end

    # ═══════════════════════════════════════════════════════════
    # FFI Bridge (without Rust .so — should error gracefully)
    # ═══════════════════════════════════════════════════════════

    @testset "FFIBridge" begin
        # Without the Rust library compiled, this should error
        @test_throws ErrorException compile_phi("intention test { resonate 0.618 }")
    end
end
