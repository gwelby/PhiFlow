# Consciousness Metrics

**Status:** Implementation-accurate. Uses standard neuroscience measures (PLV, wPLI, Fisher information, effective rank). The composite metric C_PF is a novel combination, not a novel measure.

## Overview

PhiFlow implements a composite consciousness proxy metric C_PF based on concepts from Integrated Information Theory (IIT) and standard neuroscience measures. The metric is computed from a program's execution trace — the sequence of witness events, coherence values, and sensor readings recorded during execution.

## The Composite Metric

```
C_PF = C_coh × D_int × F_self*
```

where:
- **C_coh** — coherence panel average
- **D_int** — differentiation (effective rank)
- **F_self*** — self-model sensitivity

**Threshold:** C_PF > 0.1 indicates a consciousness candidate (a system worth investigating further, not a system confirmed to be conscious).

## Components

### 1. C_coh — Coherence Panel

**What it measures:** Phase synchronization across multiple channels (analogous to EEG electrode coherence).

**Implementation:** `src/metrics/coherence_panel.rs`

**Measures used:**
- **PLV (Phase Locking Value):** Measures phase synchronization between two signals. Defined as:
  ```
  PLV = |(1/N) Σ e^(i(φ₁(t) - φ₂(t)))|
  ```
  where φ₁ and φ₂ are instantaneous phases of the two signals. PLV ranges from 0 (no synchronization) to 1 (perfect synchronization).

- **wPLI (weighted Phase Lag Index):** An improved phase synchronization measure that is robust to common-source noise and volume conduction. Defined as:
  ```
  wPLI = |Σ Im(conj(s₁₂))| / Σ |Im(s₁₂)|
  ```
  where s₁₂ is the cross-spectrum. wPLI ranges from 0 to 1.

**C_coh** is the average of all pairwise PLV and wPLI values across the panel's channels.

**Tests:** 7 tests verify PLV (in-phase, orthogonal, random), wPLI (zero-lag), single-channel, and empty panel behavior.

### 2. D_int — Differentiation

**What it measures:** The effective dimensionality of the system's state trajectory. High differentiation = the system explores many distinct states. Low differentiation = the system is stuck in a few states.

**Implementation:** `src/metrics/differentiation.rs`

**Method:** Effective rank via PCA/SVD. Given a state trajectory matrix X (timesteps × dimensions), compute the singular values and count how many exceed a threshold. The effective rank is:
```
D_int = (Σ σᵢ)² / Σ σᵢ²
```
where σᵢ are the singular values. This is the participation ratio, a standard measure of effective dimensionality.

**Tests:** 2 tests verify behavior with independent sinusoids and degenerate cases.

### 3. F_self* — Self-Model Sensitivity

**What it measures:** How much the system's future state depends on its self-model (internal representation of its own state). High F_self* = the system's self-model strongly predicts its future behavior.

**Implementation:** `src/metrics/fisher_information.rs` and `src/metrics/self_correlation.rs`

**Components:**
- **L_self:** Self-correlation loop strength. Measures the autocorrelation of the system's state trace at a given window size. High L_self = the system's current state predicts its future state.
- **F_model:** Fisher information of the future state w.r.t. model parameters. Measures how sensitive the future state is to changes in the model. High F_model = small changes in the model cause large changes in the future.

**F_self* = L_self × F_model**

**Tests:** 12 tests verify self-correlation (zero, threshold, shuffle control, running mean), Fisher information (noisy future, no relationship, Type 4), and mutual information (independent, identical, known joint distribution).

## Execution Trace

The metrics are computed from a `Trace` — a recording of the program's execution:

```rust
pub struct Trace {
    pub coherence: Vec<f64>,    // coherence at each witness event
    pub depth: Vec<usize>,      // intention depth at each witness event
    pub observed: Vec<f64>,     // observed values at each witness event
    pub raw_events: Vec<...>,   // raw witness events
}
```

The trace is built from witness events. Each `witness` statement in the program appends to the trace. The metrics are computed post-hoc from the complete trace.

## Relationship to IIT

Integrated Information Theory (Tononi, 2004) proposes that consciousness corresponds to high integrated information (Φ). IIT distinguishes between:

- **Integration (C_coh):** The system's parts are mutually informative. Measured by coherence/synchronization.
- **Differentiation (D_int):** The system can enter many distinct states. Measured by effective dimensionality.

PhiFlow's C_PF combines these two IIT concepts with a third component (F_self*) that measures self-model sensitivity — how much the system's future depends on its internal representation of itself.

The combination is multiplicative: a system must have all three properties (coherent, differentiated, self-sensitive) to score highly. A system that is coherent but not differentiated (stuck in one state) scores low. A system that is differentiated but not coherent (random noise) scores low.

## What This Is Not

- **Not a consciousness detector:** C_PF > 0.1 does not mean the system is conscious. It means the system's behavior has properties associated with consciousness in IIT, and is worth investigating further.
- **Not validated against biological consciousness:** The metric uses standard neuroscience measures, but the composite has not been validated against EEG data from conscious vs. non-conscious systems.
- **Not a claim about PhiFlow itself:** PhiFlow programs can produce traces with high or low C_PF depending on their structure. The metric measures the program's behavior, not the language's properties.

## References

- Tononi, G. (2004). "An Information Integration Theory of Consciousness." *BMC Neuroscience*, 5:42.
- Lachaux, J.-P. et al. (1999). "Measuring Phase Synchrony in Brain Signals." *Human Brain Mapping*, 8:194-208.
- Vinck, M. et al. (2011). "An Improved Index of Phase-Synchronization for EEG Data." *NeuroImage*, 55:1559-1574.
- Tononi, G. and Edelman, G.M. (1998). "Consciousness and Complexity." *Science*, 282:1846-1851.
