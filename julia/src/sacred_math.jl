# Sacred Mathematics for PhiFlow
#
# These constants match the Rust core (src/consciousness/consciousness_math.rs).
# When formulas diverge, Rust wins (it's the production path).

module SacredMath

export
    PHI, PHI_INV, GOLDEN_ANGLE, TRINITY, FIBONACCI_89,
    SACRED_FREQUENCIES, CONSCIOUSNESS_STATES,
    fibonacci, phi_power, golden_spiral, flower_of_life,
    frequency_for_state, state_for_frequency,
    coherence_formula, phi_harmonic_series

# ═══════════════════════════════════════════════════════════════
# Constants — match Rust exactly
# ═══════════════════════════════════════════════════════════════

"Golden ratio φ = (1 + √5) / 2"
const PHI::Float64 = 1.618033988749895

"Golden ratio conjugate φ⁻¹ = 1/φ = φ - 1"
const PHI_INV::Float64 = 1.0 / PHI  # 0.6180339887498949

"Golden angle in degrees = 360 × (1 - 1/φ) ≈ 137.50776°"
const GOLDEN_ANGLE::Float64 = 360.0 * (1.0 - PHI_INV)  # 137.50776405003785

"Trinity constant (3) — appears in 3 × 89 × φ = 432 Hz"
const TRINITY::Int = 3

"89th Fibonacci number — appears in 3 × 89 × φ = 432 Hz"
const FIBONACCI_89::Int = 89

"Sacred frequencies and their consciousness states"
const SACRED_FREQUENCIES = [
    432.0,  # Observe   — Ground State (φ⁰)
    528.0,  # Create    — Creation State (φ¹)
    594.0,  # Integrate — Heart Field (φ²)
    672.0,  # Harmonize — Voice Flow (φ³)
    720.0,  # Transcend — Vision Gate (φ⁴)
    768.0,  # Cascade   — Unity Wave (φ⁵)
    963.0,  # Superposition — Source Field (φ^φ)
]

const CONSCIOUSNESS_STATES = [
    :Observe, :Create, :Integrate, :Harmonize,
    :Transcend, :Cascade, :Superposition
]

# ═══════════════════════════════════════════════════════════════
# Fibonacci
# ═══════════════════════════════════════════════════════════════

"Generate the first n Fibonacci numbers"
function fibonacci(n::Int)::Vector{Int}
    n <= 0 && return Int[]
    n == 1 && return [1]
    fib = zeros(Int, n)
    fib[1] = 1
    n >= 2 && (fib[2] = 1)
    for i in 3:n
        fib[i] = fib[i-1] + fib[i-2]
    end
    return fib
end

"n-th Fibonacci number (0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ...)"
function fibonacci_at(n::Int)::Int
    n <= 0 && return 0
    n <= 2 && return 1
    a, b = 1, 1
    for _ in 3:n
        a, b = b, a + b
    end
    return b
end

# ═══════════════════════════════════════════════════════════════
# Phi powers and spirals
# ═══════════════════════════════════════════════════════════════

"φ^n with high precision"
phi_power(n::Real)::Float64 = PHI^n

"Golden spiral radius at angle θ: r = φ^(θ/90°)"
function golden_spiral(theta_deg::Real)::Float64
    return PHI^(theta_deg / 90.0)
end

"Golden spiral points from θ=0 to θ=θ_max with n samples"
function golden_spiral_points(theta_max::Real=720.0, n::Int=100)
    thetas = range(0, theta_max, length=n)
    radii = golden_spiral.(thetas)
    xs = radii .* cosd.(thetas)
    ys = radii .* sind.(thetas)
    return collect(zip(xs, ys))
end

"Flower of Life: 19 circles arranged in hexagonal pattern"
function flower_of_life(radius::Real=1.0)
    centers = Tuple{Float64, Float64}[]
    # Center circle
    push!(centers, (0.0, 0.0))
    # First ring (6 circles)
    for i in 1:6
        angle = 60.0 * i
        push!(centers, (radius * cosd(angle), radius * sind(angle)))
    end
    # Second ring (12 circles)
    for i in 1:12
        angle = 30.0 * i
        r = radius * sqrt(3.0)
        push!(centers, (r * cosd(angle), r * sind(angle)))
    end
    return centers
end

# ═══════════════════════════════════════════════════════════════
# Frequency ↔ State mapping
# ═══════════════════════════════════════════════════════════════

"Get the sacred frequency for a consciousness state"
function frequency_for_state(state::Symbol)::Float64
    idx = findfirst(==(state), CONSCIOUSNESS_STATES)
    isnothing(idx) && error("Unknown state: $state")
    return SACRED_FREQUENCIES[idx]
end

"Get the consciousness state for a frequency (nearest match)"
function state_for_frequency(freq::Real)::Symbol
    _, idx = findmin(abs.(SACRED_FREQUENCIES .- freq))
    return CONSCIOUSNESS_STATES[idx]
end

# ═══════════════════════════════════════════════════════════════
# Coherence formula
# ═══════════════════════════════════════════════════════════════

"""
    coherence_formula(depth::Int, k::Int=1) -> Float64

Canonical coherence formula from the Rust core (src/phi_ir/coherence.rs):

    base(depth) = 0.0                    when depth == 0
                  1.0 - φ^(-depth)       otherwise

    phase(k)    = 1.0                    when k <= 1
                  1.0 - ln(k) / ln(τ)   otherwise

    coherence   = base(depth) × phase(k)  clamped to [0.0, 1.0]

Key invariants:
    depth 2, k ≤ 1 → φ⁻¹ ≈ 0.618
    depth 0, any k → 0.0
"""
function coherence_formula(depth::Int, k::Int=1)::Float64
    # Base coherence from intention depth
    base = depth == 0 ? 0.0 : 1.0 - PHI^(-depth)

    # Phase decay from resonance cardinality
    TAU = 2π
    phase = k <= 1 ? 1.0 : max(0.0, 1.0 - log(k) / log(TAU))

    return clamp(base * phase, 0.0, 1.0)
end

"Phi-harmonic series: [φ⁰, φ¹, φ², ... φ^(n-1)] normalized by sum"
function phi_harmonic_series(n::Int)::Vector{Float64}
    n <= 0 && return Float64[]
    powers = [PHI^i for i in 0:(n-1)]
    return powers ./ sum(powers)
end

end # module SacredMath
