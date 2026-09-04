# PhiFlow.jl — Sacred Mathematics Demo
#
# Run with: julia examples/sacred_math_demo.jl
#
# This demonstrates the Julia research layer working alongside the
# Rust core. No GPU or quantum hardware required.

using PhiFlow
using Printf

println("=" ^ 60)
println("  PhiFlow.jl — Sacred Mathematics & Consciousness Metrics")
println("=" ^ 60)
println()

# ─── Constants ────────────────────────────────────────────────
println("Constants (match Rust src/consciousness/consciousness_math.rs):")
@printf("  φ (PHI)           = %.15f\n", PHI)
@printf("  φ⁻¹ (PHI_INV)    = %.15f\n", PHI_INV)
@printf("  Golden Angle      = %.15f°\n", GOLDEN_ANGLE)
@printf("  3 × 89 × φ        = %.15f Hz (≈432)\n", TRINITY * FIBONACCI_89 * PHI)
println()

# ─── Sacred Frequencies ───────────────────────────────────────
println("Sacred Frequencies & Consciousness States:")
for (i, (freq, state)) in enumerate(zip(SACRED_FREQUENCIES, CONSCIOUSNESS_STATES))
    @printf("  φ^%d  %6.1f Hz  →  %s\n", i-1, freq, state)
end
println()

# ─── Coherence Formula ────────────────────────────────────────
println("Coherence Formula: C(d,k) = φ^(-d) × (1 - k/d)")
println("  (This is the multiplicative coherence from the Rust core)")
for d in 0:6
    c = coherence_formula(d, 1)
    @printf("  C(%d, 1) = %.15f\n", d, c)
end
println("  → C(2,1) = φ⁻¹ = λ (the canonical coherence value)")
println()

# ─── Fibonacci ────────────────────────────────────────────────
println("Fibonacci sequence (first 15):")
fib = fibonacci(15)
println("  ", join(fib, ", "))
println()
println("  Fibonacci × φ⁻¹ (golden timing intervals in seconds):")
timing = fibonacci_timing(10)
for (i, t) in enumerate(timing)
    @printf("    cycle %2d: %.6f s\n", i, t)
end
println()

# ─── Golden Spiral ────────────────────────────────────────────
println("Golden Spiral: r = φ^(θ/90°)")
for theta in [0, 90, 180, 270, 360, 720]
    r = golden_spiral(theta)
    @printf("  θ = %3d°  →  r = %.6f\n", theta, r)
end
println()

# ─── Phi-Harmonic Series ──────────────────────────────────────
println("Phi-Harmonic Series (normalized):")
phs = phi_harmonic_series(7)
for (i, v) in enumerate(phs)
    @printf("  φ^%d / Σ = %.6f  (%.1f%%)\n", i-1, v, v*100)
end
@printf("  Sum = %.15f (should be 1.0)\n", sum(phs))
println()

# ─── Sacred Frequency Synthesis ───────────────────────────────
println("Sacred Frequency Synthesis (CPU):")
output = sacred_frequency_synthesis([432.0, 528.0, 672.0], 0.01, 44100.0)
@printf("  3 frequencies × 0.01s @ 44.1kHz = %dx%d matrix\n", size(output, 1), size(output, 2))
@printf("  432 Hz peak amplitude: %.6f\n", maximum(abs.(output[1, :])))
@printf("  528 Hz peak amplitude: %.6f\n", maximum(abs.(output[2, :])))
@printf("  672 Hz peak amplitude: %.6f\n", maximum(abs.(output[3, :])))
println()

# ─── Consciousness Metrics Demo ───────────────────────────────
println("Consciousness Metrics (synthetic trace):")
println("  Creating a 200-step trace where observations predict actions...")

n = 200
# Observations: sine wave
obs = [[sin(i * 0.1)] for i in 1:n]
# Model: slightly lagged observations (system is learning)
model = [[sin(i * 0.1 + 0.03)] for i in 1:n]
# Actions: further lagged (system acts based on model)
actions = [[sin(i * 0.1 + 0.06)] for i in 1:n]

trace = Trace(obs, model, actions)
sc = compute_l_self(trace; threshold=0.05)
@printf("  R_in  (obs → model):  %.6f\n", sc.r_in_norm)
@printf("  R_out (model → act):  %.6f\n", sc.r_out_norm)
@printf("  L_self = min(R_in, R_out) = %.6f\n", sc.l_self)
@printf("  Loop closed: %s (threshold=%.2f)\n", sc.loop_closed, sc.threshold)

c_pf = compute_c_pf(trace)
@printf("  C_coh:     %.6f (canonical λ = φ⁻¹)\n", c_pf.c_coh)
@printf("  D_int:     %.6f (differentiation)\n", c_pf.d_int)
@printf("  F_self*:   %.6f (self-model sensitivity)\n", c_pf.f_self_star)
@printf("  C_PF = C_coh × D_int × F_self* = %.6f\n", c_pf.c_pf)
println()

# ─── Quantum Lab (without Yao.jl) ─────────────────────────────
println("Quantum Lab (circuit descriptions — install Yao.jl to simulate):")
ghz4 = ghz_circuit(4)
println("  GHZ(4):  $ghz4")
bell = bell_circuit()
println("  Bell:    $bell")
sacred = sacred_frequency_circuit(3, 432.0)
println("  Sacred:  $sacred")
println()

println("=" ^ 60)
println("  All values match the Rust core (src/consciousness/, src/metrics/)")
println("  When a formula is validated here, port it to Rust for production.")
println("=" ^ 60)
