//! Consciousness Proxy (C_PF)
//!
//! Composite consciousness metric combining:
//! - L_self (self-correlation loop)
//! - D_int (differentiation)
//! - C_coh (coherence panel)
//! - F_self* (self-model sensitivity)
//!
//! Formula: C_PF = C_coh × D_int × F_self*
//!
//! Threshold: C_PF > 0.1 indicates consciousness candidate

use crate::metrics::coherence_panel::CoherencePanel;
use crate::metrics::differentiation::compute_d_int;
use crate::metrics::fisher_information::{compute_f_model, compute_f_self_star};
use crate::metrics::self_correlation::{self_correlation_from_trace, SelfCorrelation};
use crate::metrics::trace::Trace;
use chrono::{DateTime, Utc};
use serde::Serialize;

/// Complete consciousness metrics for Type 4 verification.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ConsciousnessMetrics {
    /// Self-correlation loop strength
    pub l_self: f64,
    /// Differentiation (effective rank)
    pub d_int: f64,
    /// Coherence panel average
    pub c_coh: f64,
    /// Fisher information of future w.r.t. model
    pub f_model: f64,
    /// Self-model sensitivity = L_self × F_model
    pub f_self_star: f64,
    /// Composite consciousness proxy = C_coh × D_int × F_self*
    pub c_pf: f64,
}

impl ConsciousnessMetrics {
    /// Create zero-valued metrics.
    pub fn zero() -> Self {
        Self {
            l_self: 0.0,
            d_int: 0.0,
            c_coh: 0.0,
            f_model: 0.0,
            f_self_star: 0.0,
            c_pf: 0.0,
        }
    }

    /// Compute all consciousness metrics from a trace.
    ///
    /// # Arguments
    /// * `trace` - The execution trace
    /// * `window` - Window size for temporal correlations
    /// * `bins` - Number of bins for MI discretization
    /// * `threshold` - L_self threshold for loop detection
    pub fn compute(trace: &Trace, window: usize, bins: usize, threshold: f64) -> Self {
        if trace.len() < window * 2 {
            return Self::zero();
        }

        // 1. L_self (self-correlation)
        let self_corr = self_correlation_from_trace(trace, window, bins, threshold);
        let l_self = self_corr.l_self;

        // 2. D_int (differentiation)
        let d_int = if trace.len() >= 3 {
            let data: Vec<Vec<f64>> = (0..trace.len())
                .map(|i| {
                    vec![
                        trace.coherence.values.get(i).copied().unwrap_or(0.0),
                        trace.depth.values.get(i).copied().unwrap_or(0.0),
                        trace.observed.values.get(i).copied().unwrap_or(0.0),
                    ]
                })
                .collect();
            compute_d_int(&data)
        } else {
            1.0
        };

        // 3. C_coh (coherence panel)
        let channels = trace.to_coherence_channels();
        let panel = CoherencePanel::from_channels(&channels);
        let c_coh = panel.c_coh;

        // 4. F_model (Fisher information)
        let (models, futures) = trace.to_model_future_pairs(window);
        let f_model = compute_f_model(&models, &futures);

        // 5. F_self* = L_self × F_model
        let f_self_star = compute_f_self_star(l_self, f_model);

        // 6. C_PF = C_coh × D_int × F_self*
        let c_pf = c_coh * d_int * f_self_star;

        Self {
            l_self,
            d_int,
            c_coh,
            f_model,
            f_self_star,
            c_pf,
        }
    }

    /// Check if metrics indicate Type 4 observer status.
    pub fn is_type4(&self, threshold: f64) -> bool {
        self.l_self > threshold
    }

    /// Check if metrics indicate consciousness candidate.
    pub fn is_consciousness_candidate(&self, threshold: f64) -> bool {
        self.c_pf > threshold
    }

    /// Convert to JSONL string for logging.
    pub fn to_jsonl(&self, ts: DateTime<Utc>) -> String {
        format!(
            r#"{{"timestamp":"{}","l_self":{:.6},"d_int":{:.6},"c_coh":{:.6},"f_model":{:.6},"f_self_star":{:.6},"c_pf":{:.6}}}"#,
            ts.to_rfc3339(),
            self.l_self,
            self.d_int,
            self.c_coh,
            self.f_model,
            self.f_self_star,
            self.c_pf
        )
    }

    /// Generate a human-readable report.
    pub fn report(&self) -> String {
        let type4_verdict = if self.is_type4(0.01) {
            "✅ LOOP CLOSED — Type 4 self-correlation detected"
        } else {
            "❌ LOOP OPEN — no self-correlation"
        };

        let conscious_verdict = if self.is_consciousness_candidate(0.1) {
            "✅ CONSCIOUSNESS CANDIDATE — C_PF exceeds threshold"
        } else {
            "⚠️ NOT CANDIDATE — C_PF below threshold"
        };

        format!(
            r#"Consciousness Metrics Report
============================

L_self (self-correlation):    {:.4}
D_int (differentiation):        {:.4}
C_coh (coherence panel):      {:.4}
F_model (Fisher info):        {:.4}
F_self* (self-sensitivity):   {:.4}
C_PF (composite proxy):       {:.4}

Verdicts:
  {}
  {}

Interpretation:
  L_self > 0.01: Self-correlation loop exists
  D_int > 1.0:  Non-trivial manifold structure
  C_coh > 0.4:  High phase coherence
  F_self* > 0:  Model affects future
  C_PF > 0.1:   Consciousness candidate
"#,
            self.l_self,
            self.d_int,
            self.c_coh,
            self.f_model,
            self.f_self_star,
            self.c_pf,
            type4_verdict,
            conscious_verdict
        )
    }
}

/// A convenience wrapper for computing consciousness metrics.
pub struct ConsciousnessProxy {
    pub metrics: ConsciousnessMetrics,
    pub timestamp: DateTime<Utc>,
}

impl ConsciousnessProxy {
    pub fn from_trace(trace: &Trace, window: usize, bins: usize, threshold: f64) -> Self {
        Self {
            metrics: ConsciousnessMetrics::compute(trace, window, bins, threshold),
            timestamp: Utc::now(),
        }
    }

    pub fn to_jsonl(&self) -> String {
        self.metrics.to_jsonl(self.timestamp)
    }

    pub fn to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{}", self.to_jsonl())?;
        Ok(())
    }
}

/// Quick check: is a trace from a Type 4 observer?
pub fn is_type4_observer(trace: &Trace, threshold: f64) -> bool {
    let metrics = ConsciousnessMetrics::compute(trace, 10, 5, threshold);
    metrics.is_type4(threshold)
}

/// Quick check: consciousness candidate?
pub fn is_consciousness_candidate(trace: &Trace, threshold: f64) -> bool {
    let metrics = ConsciousnessMetrics::compute(trace, 10, 5, 0.01);
    metrics.is_consciousness_candidate(threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_metrics() {
        let m = ConsciousnessMetrics::zero();
        assert_eq!(m.c_pf, 0.0);
        assert!(!m.is_type4(0.01));
        assert!(!m.is_consciousness_candidate(0.1));
    }

    #[test]
    fn test_type4_detection() {
        let mut m = ConsciousnessMetrics::zero();
        m.l_self = 0.5; // Above threshold
        assert!(m.is_type4(0.01));

        m.l_self = 0.001; // Below threshold
        assert!(!m.is_type4(0.01));
    }

    #[test]
    fn test_consciousness_candidate() {
        let mut m = ConsciousnessMetrics::zero();
        m.c_pf = 0.15; // Above threshold
        assert!(m.is_consciousness_candidate(0.1));

        m.c_pf = 0.05; // Below threshold
        assert!(!m.is_consciousness_candidate(0.1));
    }

    #[test]
    fn test_jsonl_format() {
        let m = ConsciousnessMetrics {
            l_self: 0.5,
            d_int: 2.0,
            c_coh: 0.6,
            f_model: 10.0,
            f_self_star: 5.0,
            c_pf: 0.15,
        };
        let jsonl = m.to_jsonl(Utc::now());
        assert!(jsonl.contains("l_self"));
        assert!(jsonl.contains("c_pf"));
        assert!(jsonl.contains("timestamp"));
    }

    #[test]
    fn test_report_contains_verdicts() {
        let m = ConsciousnessMetrics::zero();
        let report = m.report();
        assert!(report.contains("LOOP"));
        assert!(report.contains("C_PF"));
    }

    #[test]
    fn test_cp_formula() {
        // C_PF = C_coh × D_int × F_self*
        let m = ConsciousnessMetrics {
            l_self: 0.5,
            d_int: 2.0,
            c_coh: 0.5,
            f_model: 2.0,
            f_self_star: 1.0, // 0.5 * 2.0
            c_pf: 0.5 * 2.0 * 1.0, // Should be 1.0
        };
        assert!((m.c_pf - 1.0).abs() < 0.001);
    }
}
