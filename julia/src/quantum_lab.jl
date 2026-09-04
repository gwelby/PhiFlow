# Quantum Lab — Circuit Prototyping with Yao.jl
#
# This module is lazy-loaded: if Yao.jl is not installed, the functions
# return informative errors. When Yao is available, you can prototype
# quantum circuits before porting them to the Rust OpenQASM emitter.
#
# What this does that Rust doesn't:
#   - Interactive circuit construction and simulation
#   - Parameterized circuit families (GHZ, sacred frequency, phi-harmonic)
#   - Quick measurement statistics and visualization
#   - Research-grade quantum state tomography
#
# To enable:
#   using Pkg; Pkg.add("Yao")
#
# The Rust core has its own simulator (src/quantum/simulator.rs) for
# production. This is for research — when a circuit family is validated
# here, the OpenQASM emission path in Rust is the production path.

module QuantumLab

using ..SacredMath: PHI, PHI_INV, SACRED_FREQUENCIES, CONSCIOUSNESS_STATES

export
    ghz_circuit, bell_circuit, sacred_frequency_circuit,
    phi_harmonic_circuit, measure_circuit,
    ghz_coherence_scaling

# ─── Circuit builders (return Yao chains when available) ──────

"""
    ghz_circuit(n_qubits::Int)

Build a GHZ state: H on qubit 1, then CNOT chain.
Returns a Yao circuit (requires Yao.jl) or a description string.
"""
function ghz_circuit(n_qubits::Int)
    try
        @eval using Yao
        @eval using Yao.Blocks

        # Build: H(1) → CNOT(1,2) → CNOT(2,3) → ... → CNOT(n-1,n)
        chain = Yao.Blocks.chain(n_qubits)
        push!(chain, Yao.Blocks.h(1))
        for i in 1:(n_qubits-1)
            push!(chain, Yao.Blocks.cnot(i, i+1))
        end
        return chain
    catch
        return "GHZ($n_qubits): H(1) → CNOT(1,2) → ... → CNOT($(n_qubits-1),$n_qubits) [install Yao.jl to simulate]"
    end
end

"""
    bell_circuit()

Build a Bell state (2-qubit GHZ): H(1) → CNOT(1,2)
"""
function bell_circuit()
    return ghz_circuit(2)
end

"""
    sacred_frequency_circuit(n_qubits::Int, frequency::Float64)

Build a circuit that encodes a sacred frequency into quantum rotations.
Each qubit gets an RY rotation proportional to the frequency / phi.

This is the Julia prototype of the Rust QuantumGate::SacredFrequency.
"""
function sacred_frequency_circuit(n_qubits::Int, frequency::Float64)
    try
        @eval using Yao
        @eval using Yao.Blocks

        # Angle = 2π × frequency / (PHI × max_freq)
        angle = 2π * frequency / (PHI * maximum(SACRED_FREQUENCIES))

        chain = Yao.Blocks.chain(n_qubits)
        for q in 1:n_qubits
            push!(chain, Yao.Blocks.ry(q, angle * PHI^(q-1)))
        end
        return chain
    catch
        angle = 2π * frequency / (PHI * maximum(SACRED_FREQUENCIES))
        return "SacredFreq($n_qubits, $(frequency)Hz): RY(q, $(angle) × φ^(q-1)) [install Yao.jl to simulate]"
    end
end

"""
    phi_harmonic_circuit(n_qubits::Int, phi_power::Int)

Build a circuit with phi-harmonic rotations.
Each qubit q gets RY(π × φ^(phi_power - q + 1)).

This is the Julia prototype of the Rust QuantumGate::PhiHarmonic.
"""
function phi_harmonic_circuit(n_qubits::Int, phi_power::Int)
    try
        @eval using Yao
        @eval using Yao.Blocks

        chain = Yao.Blocks.chain(n_qubits)
        for q in 1:n_qubits
            angle = π * PHI^(phi_power - q + 1)
            push!(chain, Yao.Blocks.ry(q, angle))
        end
        return chain
    catch
        return "PhiHarmonic($n_qubits, φ^$phi_power): RY(q, π × φ^($phi_power - q + 1)) [install Yao.jl to simulate]"
    end
end

# ─── Measurement and analysis ─────────────────────────────────

"""
    measure_circuit(circuit, n_shots::Int=1024)

Measure a circuit n_shots times. Returns a dictionary of bitstring → count.
Requires Yao.jl.
"""
function measure_circuit(circuit, n_shots::Int=1024)
    try
        @eval using Yao

        # Get the number of qubits from the circuit
        n = Yao.nqubits(circuit)

        # Run shots
        results = Dict{String, Int}()
        for _ in 1:n_shots
            # Apply circuit to zero state, measure all qubits
            state = Yao.zero_state(n)
            apply!(state, circuit)
            bits = Yao.measure!(state, 1:n)
            key = join(bits)
            results[key] = get(results, key, 0) + 1
        end
        return results
    catch e
        return Dict("error" => "Yao.jl required: $(e)")
    end
end

# ─── GHZ coherence scaling study ──────────────────────────────

"""
    ghz_coherence_scaling(max_qubits::Int, n_shots::Int=4096)

Study how GHZ coherence scales with qubit count.
This reproduces the C-26 claim: GHZ coherence on Heron-R2.

Returns a table of (n_qubits, coherence) pairs.
Coherence = 2 × P(00...0) - 1 (ideal GHZ has P = 0.5 → coherence = 1.0).

NOTE: This uses the local simulator, not IBM hardware.
The real GHZ scaling data is in reports/GHZ_SCALING_2026-07-10.md.
"""
function ghz_coherence_scaling(max_qubits::Int, n_shots::Int=4096)
    results = Tuple{Int, Float64}[]

    for n in 2:max_qubits
        circuit = ghz_circuit(n)
        if circuit isa String
            push!(results, (n, NaN))
            continue
        end

        counts = measure_circuit(circuit, n_shots)
        if haskey(counts, "error")
            push!(results, (n, NaN))
            continue
        end

        # GHZ coherence: 2 × (P(00..0) + P(11..1)) - 1
        p_0 = get(counts, "0"^n, 0) / n_shots
        p_1 = get(counts, "1"^n, 0) / n_shots
        coherence = 2 * (p_0 + p_1) - 1
        push!(results, (n, coherence))
    end

    return results
end

end # module QuantumLab
