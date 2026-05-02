//! PhiFlow Consciousness Metrics Module
//!
//! Implements the PF consciousness metric program for Type 4 observer verification.
//! All metrics are read-only observers of the existing VmState.witness_log and
//! resonance_events structures—no new data schemas, zero migration risk.

pub mod coherence_panel;
pub mod consciousness_proxy;
pub mod differentiation;
pub mod fisher_information;
pub mod mutual_information;
pub mod self_correlation;
pub mod trace;

// Re-export main types for convenience
pub use coherence_panel::CoherencePanel;
pub use consciousness_proxy::{ConsciousnessMetrics, ConsciousnessProxy};
pub use differentiation::compute_d_int;
pub use fisher_information::{compute_f_model, compute_f_self_star};
pub use mutual_information::{conditional_mi, mi_from_samples, normalized_mi, shannon_mi};
pub use self_correlation::{self_correlation_from_trace, SelfCorrelation};
pub use trace::{Trace, TraceChannel};
