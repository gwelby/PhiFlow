//! Coherence Panel (PLV + wPLI)
//!
//! Implements Phase Locking Value (PLV) and weighted Phase Lag Index (wPLI)
//! for multi-channel coherence analysis.
//!
//! PLV: measures phase synchronization (sensitive to volume conduction)
//! wPLI: suppresses zero-lag correlations (more robust)
//!
//! This panel SUPPLEMENTS the canonical PhiFlow coherence formula—it does
//! not replace it. Both are used: canonical for runtime, panel for C_PF.

use num_complex::Complex;
use rustfft::FftPlanner;

/// A panel of coherence metrics between multiple signal channels.
#[derive(Debug, Clone)]
pub struct CoherencePanel {
    /// Phase Locking Value matrix (symmetric, 0..1)
    pub plv_matrix: Vec<Vec<f64>>,
    /// weighted Phase Lag Index matrix (symmetric, 0..1)
    pub wpli_matrix: Vec<Vec<f64>>,
    /// Average coherence score (C_coh proxy)
    pub c_coh: f64,
    /// Number of channels
    pub n_channels: usize,
}

impl CoherencePanel {
    /// Create an empty coherence panel.
    pub fn new(n: usize) -> Self {
        Self {
            plv_matrix: vec![vec![0.0; n]; n],
            wpli_matrix: vec![vec![0.0; n]; n],
            c_coh: 0.0,
            n_channels: n,
        }
    }

    /// Compute PLV between two signals.
    ///
    /// PLV measures consistency of phase differences across time.
    /// PLV = |<exp(i * (phase1 - phase2))>|
    pub fn compute_plv(signal1: &[f64], signal2: &[f64]) -> f64 {
        if signal1.len() != signal2.len() || signal1.len() < 2 {
            return 0.0;
        }

        let phase1 = analytic_phase(signal1);
        let phase2 = analytic_phase(signal2);

        // Compute phase difference vector sum
        let mut sum = Complex::new(0.0, 0.0);
        for (p1, p2) in phase1.iter().zip(phase2.iter()) {
            let diff = p1 - p2;
            sum += Complex::new(diff.cos(), diff.sin());
        }

        (sum / signal1.len() as f64).norm()
    }

    /// Compute weighted Phase Lag Index (wPLI).
    ///
    /// wPLI suppresses zero-lag correlations by weighting by the imaginary
    /// component of the cross-spectrum. This makes it robust to volume
    /// conduction artifacts.
    pub fn compute_wpli(signal1: &[f64], signal2: &[f64]) -> f64 {
        if signal1.len() != signal2.len() || signal1.len() < 2 {
            return 0.0;
        }

        let phase1 = analytic_phase(signal1);
        let phase2 = analytic_phase(signal2);

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (p1, p2) in phase1.iter().zip(phase2.iter()) {
            let diff = p1 - p2;
            let imag = diff.sin(); // sin(Δφ) is proportional to Im[S_xy]
            numerator += imag.abs() * imag.signum();
            denominator += imag.abs();
        }

        if denominator > 0.0 {
            (numerator / denominator).abs()
        } else {
            0.0
        }
    }

    /// Build coherence panel from multiple channels.
    pub fn from_channels(channels: &[Vec<f64>]) -> Self {
        let n = channels.len();
        if n == 0 {
            return Self::new(0);
        }

        let mut panel = Self::new(n);

        for i in 0..n {
            for j in (i + 1)..n {
                let plv = Self::compute_plv(&channels[i], &channels[j]);
                let wpli = Self::compute_wpli(&channels[i], &channels[j]);

                panel.plv_matrix[i][j] = plv;
                panel.plv_matrix[j][i] = plv;
                panel.wpli_matrix[i][j] = wpli;
                panel.wpli_matrix[j][i] = wpli;
            }
        }

        // Compute average C_coh
        panel.compute_c_coh();
        panel
    }

    /// Compute average coherence proxy (C_coh).
    ///
    /// Averages upper triangle of (PLV + wPLI) / 2.
    fn compute_c_coh(&mut self) {
        let n = self.n_channels;
        if n < 2 {
            self.c_coh = 0.0;
            return;
        }

        let mut plv_sum = 0.0;
        let mut wpli_sum = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                plv_sum += self.plv_matrix[i][j];
                wpli_sum += self.wpli_matrix[i][j];
                count += 1;
            }
        }

        if count > 0 {
            self.c_coh = (plv_sum + wpli_sum) / (2.0 * count as f64);
        } else {
            self.c_coh = 0.0;
        }
    }

    /// Get coherence between two specific channels.
    pub fn get_plv(&self, i: usize, j: usize) -> f64 {
        if i < self.n_channels && j < self.n_channels {
            self.plv_matrix[i][j]
        } else {
            0.0
        }
    }

    /// Get wPLI between two specific channels.
    pub fn get_wpli(&self, i: usize, j: usize) -> f64 {
        if i < self.n_channels && j < self.n_channels {
            self.wpli_matrix[i][j]
        } else {
            0.0
        }
    }
}

/// Compute analytic signal via Hilbert transform using FFT.
/// Returns instantaneous phase for each sample.
fn analytic_phase(signal: &[f64]) -> Vec<f64> {
    if signal.len() < 2 {
        return vec![0.0; signal.len()];
    }

    // Zero-pad to power of 2 for FFT efficiency
    let n_original = signal.len();
    let n = n_original.next_power_of_two();

    // Copy signal to complex buffer
    let mut buffer: Vec<Complex<f64>> = signal
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    buffer.resize(n, Complex::new(0.0, 0.0));

    // Forward FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Create analytic signal: double positive frequencies, zero negative
    let mut analytic = buffer.clone();
    for i in 1..(n / 2) {
        analytic[i] *= 2.0; // Double positive frequencies
    }
    for i in ((n / 2) + 1)..n {
        analytic[i] = Complex::new(0.0, 0.0); // Zero negative frequencies
    }
    if n % 2 == 0 {
        // Nyquist frequency (n/2) stays the same
    }

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut analytic);

    // Extract phase from analytic signal
    let mut phases = Vec::with_capacity(n_original);
    for i in 0..n_original {
        phases.push(analytic[i].arg());
    }

    phases
}

/// Simplified phase extraction without full Hilbert transform.
/// Uses arctan of signal derivative as a proxy.
#[allow(dead_code)]
fn simple_phase(signal: &[f64]) -> Vec<f64> {
    if signal.len() < 2 {
        return vec![0.0; signal.len()];
    }

    let mut phases = Vec::with_capacity(signal.len());
    phases.push(0.0); // First sample has no derivative

    for i in 1..signal.len() {
        let diff = signal[i] - signal[i - 1];
        phases.push(diff.atan());
    }

    phases
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plv_in_phase() {
        // Two in-phase sinusoids should have PLV ≈ 1
        let s1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let s2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();

        let plv = CoherencePanel::compute_plv(&s1, &s2);
        assert!(plv > 0.95, "In-phase sinusoids should have PLV > 0.95, got {}", plv);
    }

    #[test]
    fn test_plv_orthogonal() {
        // Sine and cosine (90° phase shift) should have lower PLV
        let s1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let s2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();

        let plv = CoherencePanel::compute_plv(&s1, &s2);
        // Orthogonal signals should have PLV ≈ 1 (consistent phase difference)
        assert!(plv > 0.8, "Orthogonal sinusoids should still have high PLV, got {}", plv);
    }

    #[test]
    fn test_plv_random() {
        // Two random signals should have PLV ≈ 1/sqrt(N) (low)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let s1: Vec<f64> = (0..200).map(|_| rng.gen::<f64>()).collect();
        let s2: Vec<f64> = (0..200).map(|_| rng.gen::<f64>()).collect();

        let plv = CoherencePanel::compute_plv(&s1, &s2);
        assert!(plv < 0.3, "Random signals should have low PLV, got {}", plv);
    }

    #[test]
    fn test_wpli_zero_lag() {
        // Identical zero-lag signals should be suppressed by wPLI.
        let s1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let s2 = s1.clone();

        let wpli = CoherencePanel::compute_wpli(&s1, &s2);
        assert!(wpli < 0.05, "Zero-lag signals should have low wPLI, got {}", wpli);
    }

    #[test]
    fn test_panel_from_channels() {
        let channels = vec![
            (0..100).map(|i| (i as f64 * 0.1).sin()).collect(),
            (0..100).map(|i| (i as f64 * 0.1).cos()).collect(),
            (0..100).map(|i| (i as f64 * 0.05).sin()).collect(),
        ];

        let panel = CoherencePanel::from_channels(&channels);
        assert_eq!(panel.n_channels, 3);
        assert!(panel.c_coh > 0.0, "Panel should have some coherence");
        println!("Panel C_coh: {}", panel.c_coh);
    }

    #[test]
    fn test_empty_panel() {
        let panel = CoherencePanel::from_channels(&[]);
        assert_eq!(panel.n_channels, 0);
        assert_eq!(panel.c_coh, 0.0);
    }

    #[test]
    fn test_single_channel() {
        let channels = vec![(0..10).map(|i| i as f64).collect()];
        let panel = CoherencePanel::from_channels(&channels);
        assert_eq!(panel.n_channels, 1);
        // Single channel has no pairs, so C_coh = 0
        assert_eq!(panel.c_coh, 0.0);
    }

    #[test]
    fn test_analytic_phase() {
        // Test that analytic_phase produces reasonable output
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let phases = analytic_phase(&signal);
        assert_eq!(phases.len(), signal.len());
        // Phase should be between -π and π
        for &p in &phases {
            assert!(p >= -std::f64::consts::PI && p <= std::f64::consts::PI);
        }
    }
}
