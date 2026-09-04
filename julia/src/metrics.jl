# Consciousness Metrics Research Module
#
# This mirrors the Rust metrics (src/metrics/) but is designed for
# rapid experimentation. When a formula is validated here, it gets
# ported to Rust.
#
# Key metrics:
#   L_self  = min(R_in, R_out)  — self-correlation loop strength
#   F_model = Fisher information of future w.r.t. model state
#   C_PF    = C_coh × D_int × F_self*  — consciousness proxy
#
# Type 4 observer: a system whose model state at time t both
# depends on past observations AND predicts future behavior.

module Metrics

using Statistics
using LinearAlgebra

export
    SelfCorrelation, FisherInfo, ConsciousnessProxy,
    compute_l_self, compute_f_model, compute_c_pf,
    mutual_information, normalized_mi,
    differentiate, Trace

# ═══════════════════════════════════════════════════════════════
# Trace — a time series of observations/model/actions
# ═══════════════════════════════════════════════════════════════

"""
A trace is a sequence of time steps, each with:
- observations: what the system sensed
- model_state:  the system's internal model
- actions:      what the system did

For Type 4 analysis, we need all three channels.
"""
struct Trace
    observations::Vector{Vector{Float64}}
    model_states::Vector{Vector{Float64}}
    actions::Vector{Vector{Float64}}
end

"Length of a trace"
Base.length(t::Trace) = length(t.observations)

"Create a simple trace from a single channel (treats it as observations)"
function Trace(channel::Vector{Vector{Float64}})
    return Trace(channel, copy(channel), copy(channel))
end

# ═══════════════════════════════════════════════════════════════
# Mutual Information (normalized)
# ═══════════════════════════════════════════════════════════════

"""
Estimate mutual information between two scalar sequences using
histogram-based method. Returns normalized MI (0 to 1).

R_in  = I(past_observations → model_state)
R_out = I(model_state → future_actions)
"""
function mutual_information(x::Vector{Float64}, y::Vector{Float64}; n_bins::Int=10)::Float64
    n = length(x)
    n != length(y) && error("Length mismatch")
    n < 2 && return 0.0

    # Discretize into bins
    x_bins = digitize(x, n_bins)
    y_bins = digitize(y, n_bins)

    # Joint and marginal distributions
    p_xy = zeros(Float64, n_bins, n_bins)
    for i in 1:n
        p_xy[x_bins[i], y_bins[i]] += 1.0
    end
    p_xy ./= n
    p_x = sum(p_xy, dims=2)
    p_y = sum(p_xy, dims=1)

    # MI = Σ p(x,y) × log( p(x,y) / (p(x) × p(y)) )
    mi = 0.0
    for i in 1:n_bins, j in 1:n_bins
        if p_xy[i,j] > 0 && p_x[i] > 0 && p_y[j] > 0
            mi += p_xy[i,j] * log2(p_xy[i,j] / (p_x[i] * p_y[j]))
        end
    end

    return mi
end

"Normalized MI: NMI = MI / sqrt(H(x) × H(y))"
function normalized_mi(x::Vector{Float64}, y::Vector{Float64}; n_bins::Int=10)::Float64
    n = length(x)
    n < 2 && return 0.0

    mi = mutual_information(x, y; n_bins=n_bins)

    # Entropies
    x_bins = digitize(x, n_bins)
    y_bins = digitize(y, n_bins)
    p_x = [count(==(b), x_bins) for b in 1:n_bins] ./ n
    p_y = [count(==(b), y_bins) for b in 1:n_bins] ./ n

    h_x = -sum(p .* log2.(p) for p in p_x if p > 0)
    h_y = -sum(p .* log2.(p) for p in p_y if p > 0)

    h_x <= 0 || h_y <= 0 && return 0.0
    return mi / sqrt(h_x * h_y)
end

# Helper: discretize into bins
function digitize(x::Vector{Float64}, n_bins::Int)::Vector{Int}
    lo, hi = extrema(x)
    lo == hi && return ones(Int, length(x))
    bins = clamp.(Int.(floor.((x .- lo) ./ (hi - lo) .* n_bins)) .+ 1, 1, n_bins)
    return bins
end

# ═══════════════════════════════════════════════════════════════
# Self-Correlation (L_self)
# ═══════════════════════════════════════════════════════════════

"""
Self-correlation metrics for Type 4 observer verification.

  R_in  = I_dir(past_observations → model_state)
  R_out = I_dir(model_state → future_behavior | current_obs)
  L_self = min(R_in_norm, R_out_norm)

If either leg is zero, the self-correlation loop is broken → not Type 4.
"""
struct SelfCorrelation
    r_in_norm::Float64    # Normalized MI: past observations → current model
    r_out_norm::Float64   # Normalized MI: current model → future behavior
    l_self::Float64       # Self-correlation loop strength: min(R_in, R_out)
    loop_closed::Bool     # L_self > threshold
    threshold::Float64
end

"""
    compute_l_self(trace::Trace; threshold=0.1, lag=1) -> SelfCorrelation

Compute self-correlation from a trace.
- R_in:  MI(observations[1:end-lag], model_states[lag+1:end])
- R_out: MI(model_states[1:end-lag], actions[lag+1:end])
"""
function compute_l_self(trace::Trace; threshold::Float64=0.1, lag::Int=1)::SelfCorrelation
    n = length(trace)
    n <= lag + 1 && return SelfCorrelation(0.0, 0.0, 0.0, false, threshold)

    # R_in: past observations → current model
    past_obs = [trace.observations[i][1] for i in 1:(n-lag)]
    curr_model = [trace.model_states[i+lag][1] for i in 1:(n-lag)]
    r_in = normalized_mi(past_obs, curr_model)

    # R_out: current model → future actions
    curr_model2 = [trace.model_states[i][1] for i in 1:(n-lag)]
    future_act = [trace.actions[i+lag][1] for i in 1:(n-lag)]
    r_out = normalized_mi(curr_model2, future_act)

    l_self = min(r_in, r_out)
    return SelfCorrelation(r_in, r_out, l_self, l_self > threshold, threshold)
end

# ═══════════════════════════════════════════════════════════════
# Fisher Information (F_model)
# ═══════════════════════════════════════════════════════════════

"""
Fisher information of future trajectory w.r.t. model state.
F = E[(∂log p(future|model) / ∂model)²]

Estimated via finite differences: how sharply does the future
change when the model state is perturbed?
"""
struct FisherInfo
    f_model::Float64       # Fisher information of future w.r.t. model
    f_self_star::Float64   # L_self × F_model (self-model sensitivity)
end

function compute_f_model(model_states::Vector{Vector{Float64}},
                         future_trajectories::Vector{Vector{Float64}})::Float64
    n = length(model_states)
    n != length(future_trajectories) && error("Length mismatch")
    n < 2 && return 0.0

    epsilon = 1e-6
    fisher_sum = 0.0

    for i in 1:n
        m = model_states[i]
        f = future_trajectories[i]
        isempty(m) && continue
        isempty(f) && continue

        # Finite difference: perturb model, see how future changes
        for j in 1:min(length(m), length(f))
            m_plus = copy(m)
            m_plus[j] += epsilon
            # In a real system, we'd compute p(future | m_plus)
            # Here we approximate with the sensitivity of f to m
            df_dm = (f[j] - m[j]) / (m[j] + epsilon - m[j] + 1e-30)
            fisher_sum += df_dm^2
        end
    end

    return fisher_sum / n
end

"Compute F_self* = L_self × F_model"
function compute_f_self_star(l_self::Float64, f_model::Float64)::Float64
    return l_self * f_model
end

# ═══════════════════════════════════════════════════════════════
# Consciousness Proxy (C_PF)
# ═══════════════════════════════════════════════════════════════

"""
Consciousness proxy metric:
    C_PF = C_coh × D_int × F_self*

Where:
    C_coh     = coherence (from sacred math or sensor measurement)
    D_int     = differentiation (how distinct are system states)
    F_self*   = L_self × F_model (self-model sensitivity)
"""
struct ConsciousnessProxy
    c_coh::Float64       # Coherence
    d_int::Float64       # Differentiation
    f_self_star::Float64 # Self-model sensitivity
    c_pf::Float64        # Consciousness proxy: C_coh × D_int × F_self*
end

"""
    compute_c_pf(trace::Trace; threshold=0.1) -> ConsciousnessProxy

Compute the full consciousness proxy from a trace.
"""
function compute_c_pf(trace::Trace; threshold::Float64=0.1)::ConsciousnessProxy
    sc = compute_l_self(trace; threshold=threshold)

    # F_model: how much future depends on model
    n = length(trace)
    if n > 2
        half = n ÷ 2
        f_model = compute_f_model(
            trace.model_states[1:half],
            [trace.actions[i] for i in (half+1):n]
        )
    else
        f_model = 0.0
    end

    f_self_star = compute_f_self_star(sc.l_self, f_model)

    # D_int: differentiation — variance of observations normalized by max possible
    if n > 0 && !isempty(trace.observations[1])
        all_obs = hcat(trace.observations...)
        d_int = mean(std(all_obs, dims=2))
        d_int = clamp(d_int, 0.0, 1.0)
    else
        d_int = 0.0
    end

    # C_coh: coherence — use the formula at depth 2 (canonical λ = φ⁻¹)
    c_coh = 0.618033988749895  # φ⁻¹ — replace with sensor measurement when available

    c_pf = c_coh * d_int * f_self_star

    return ConsciousnessProxy(c_coh, d_int, f_self_star, c_pf)
end

# ═══════════════════════════════════════════════════════════════
# Differentiation (D_int)
# ═══════════════════════════════════════════════════════════════

"""
Differentiation: how distinct are the system's states?
Measured as the mean normalized variance across all channels.

High D_int = system visits many distinct states
Low D_int  = system stays in one state
"""
function differentiate(data::Vector{Vector{Float64}})::Float64
    isempty(data) && return 0.0
    n = length(data)
    dim = length(data[1])

    total_var = 0.0
    for d in 1:dim
        channel = [data[i][d] for i in 1:n]
        lo, hi = extrema(channel)
        range = hi - lo
        range > 0 && (total_var += var(channel) / range^2)
    end

    return total_var / dim
end

end # module Metrics
