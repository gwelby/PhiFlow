use anyhow::Result;
use std::collections::HashMap;

const PHI_INV: f64 = 0.618033988749895;

/// Mock simulation modes for the self-correction loop demonstration.
///
/// `Decoherent` simulates a pre-correction state where the quantum system
/// has decohered (01/10 split, coherence ≈ 0.0).
///
/// `Coherent` simulates a post-correction state where the correction has
/// restored entanglement (00/11 Bell-state split, coherence ≈ 1.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockMode {
    Decoherent,
    Coherent,
}

/// Returns mock measurement counts for the given simulation mode.
///
/// - `Decoherent`: all shots in |01⟩/|10⟩ → Bell-state coherence = 0.0
/// - `Coherent`:   all shots in |00⟩/|11⟩ → Bell-state coherence = 1.0
pub fn poll_ibm_job_mock(mode: MockMode) -> HashMap<String, u64> {
    let mut counts = HashMap::new();
    match mode {
        MockMode::Decoherent => {
            counts.insert("01".to_string(), 512);
            counts.insert("10".to_string(), 488);
        }
        MockMode::Coherent => {
            counts.insert("00".to_string(), 512);
            counts.insert("11".to_string(), 488);
        }
    }
    counts
}

/// Polls an IBM Quantum job and extracts the measurement counts.
///
/// This function handles **mock mode only** — when `credential` is "MOCK_KEY"
/// or empty, it returns synthetic counts for demo purposes.
///
/// For real IBM Quantum jobs, the Rust CLI shells out to
/// `scripts/poll_ibm_real.py` (Python bridge using `qiskit_ibm_runtime`
/// with the `ibm_quantum_platform` channel). The old REST API is deprecated.
pub fn poll_ibm_job(job_id: &str, credential: &str) -> Result<HashMap<String, u64>> {
    if credential == "MOCK_KEY" || credential.is_empty() {
        // Default mock: decoherent state so the self-correction loop can fire.
        // Use `poll_ibm_job_mock(MockMode::Coherent)` for high-coherence counts.
        return Ok(poll_ibm_job_mock(MockMode::Decoherent));
    }

    // Real job polling is handled by the Python bridge (scripts/poll_ibm_real.py).
    // This function returns empty counts if called with a real credential,
    // indicating that the caller should use the Python bridge instead.
    let _ = job_id; // suppress unused warning
    Ok(HashMap::new())
}

/// Analyzes measurement distribution to calculate physical coherence.
/// Evaluates how far the state is from a maximally entangled or expected state.
///
/// Handles three cases based on the bit-width of the measurement states:
/// - **1-qubit** (states "0"/"1"): coherence = max(p0, p1) — concentration
/// - **2-qubit** (states "00".."11"): Bell-state coherence = (p00 + p11) / total
/// - **3+ qubit** (longer bitstrings): concentration = max_count / total,
///   measuring how peaked the distribution is. A maximally coherent state
///   (all shots in one state) gives 1.0; a uniform distribution gives 1/N.
pub fn calculate_coherence(counts: &HashMap<String, u64>) -> f64 {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    // Determine bit width from the longest state string
    let max_len = counts.keys().map(|s| s.len()).max().unwrap_or(0);

    match max_len {
        // 1-qubit: concentration measure
        0 | 1 => {
            let count_0 = counts.get("0").copied().unwrap_or(0);
            let count_1 = counts.get("1").copied().unwrap_or(0);
            let max_count = count_0.max(count_1);
            max_count as f64 / total as f64
        }
        // 2-qubit: Bell-state coherence (00+11 are "good", 01+10 are "bad")
        2 => {
            let count_00 = counts.get("00").copied().unwrap_or(0);
            let count_11 = counts.get("11").copied().unwrap_or(0);
            let count_01 = counts.get("01").copied().unwrap_or(0);
            let count_10 = counts.get("10").copied().unwrap_or(0);

            let good_states = count_00 + count_11;
            let bad_states = count_01 + count_10;

            if good_states + bad_states == 0 {
                // States don't match 2-bit patterns — fall back to concentration
                let max_count = counts.values().copied().max().unwrap_or(0);
                max_count as f64 / total as f64
            } else {
                good_states as f64 / total as f64
            }
        }
        // 3+ qubits: GHZ coherence — fraction of shots in the two
        // expected entangled basis states (all-0s and all-1s).
        // This is the N-qubit generalization of the Bell-state check:
        // a GHZ state (|0...0⟩ + |1...1⟩)/√2 should produce only
        // |0...0⟩ and |1...1⟩ measurements. Any other bitstring is
        // a decoherence event.
        //
        // If neither all-0s nor all-1s appears, fall back to concentration.
        _ => {
            let all_zeros = "0".repeat(max_len);
            let all_ones = "1".repeat(max_len);
            let count_00 = counts.get(&all_zeros).copied().unwrap_or(0);
            let count_11 = counts.get(&all_ones).copied().unwrap_or(0);
            let ghz_states = count_00 + count_11;

            if ghz_states > 0 {
                ghz_states as f64 / total as f64
            } else {
                // No GHZ basis states found — fall back to concentration
                let max_count = counts.values().copied().max().unwrap_or(0);
                max_count as f64 / total as f64
            }
        }
    }
}

/// Evaluates coherence against the Phi Inverse threshold (0.618).
/// If physical coherence is lower, generates a correction plan.
///
/// The correction is zero-depth RZ correction: shift existing RZ gates
/// that follow CZ gates by a backend-specific optimal angle. This corrects
/// coherent Z⊗I over-rotation on CX gates — a real error mechanism proven
/// on IBM hardware by the Crypto lab (R-35 through R-40).
///
/// Key results from the Crypto lab:
/// - R-36: RZ correction reduces FP rate 65% → 5% on simulator
/// - R-40: Zero-depth RZ correction reduces FP 10-20pp on real hardware
/// - Kingston optimal: +0.045 (U-shape confirmed across 9 angles)
/// - Fez optimal: -0.090 (opposite sign confirms coherent error model)
/// - Zero new gates, zero depth increase — the correction is free
///
/// The actual circuit correction is performed by the Python bridge
/// (`scripts/self_correction_real.py`) which applies the zero-depth
/// RZ shift and submits the corrected circuit to IBM Quantum.
pub fn generate_correction_if_needed(coherence: f64) -> Option<String> {
    if coherence < PHI_INV {
        let correction = format!(
            r#"intention "self_correction" {{
    let initial_coherence = {coherence}
    let threshold = {threshold}
    let correction_method = "zero_depth_rz"
    witness
}}"#,
            coherence = coherence,
            threshold = PHI_INV,
        );
        Some(correction)
    } else {
        None
    }
}

/// Result of a self-correction cycle.
#[derive(Debug)]
pub struct CorrectionResult {
    /// Coherence before correction was applied.
    pub initial_coherence: f64,
    /// Coherence after correction was applied (re-measured).
    pub final_coherence: f64,
    /// The PhiFlow correction code that was generated and executed.
    pub correction_source: Option<String>,
    /// Whether the correction was executed successfully.
    pub correction_executed: bool,
}

impl CorrectionResult {
    /// True if the correction improved coherence above the φ⁻¹ threshold.
    pub fn improved(&self) -> bool {
        self.final_coherence > self.initial_coherence && self.final_coherence >= PHI_INV
    }

    /// The improvement in coherence (final - initial).
    pub fn delta(&self) -> f64 {
        self.final_coherence - self.initial_coherence
    }
}

/// Runs the self-correction loop using mock counts.
///
/// This is a STRUCTURAL TEST of the loop, not a real correction.
/// The mock counts are hardcoded: decoherent before, coherent after.
/// The "improvement" is simulated.
///
/// For real quantum self-correction with actual IBM hardware, use:
///   python3.12 scripts/self_correction_real.py <n> <backend> <shots>
///
/// That script submits a real circuit, gets real counts, detects real
/// decoherence, re-routes to better qubits, and measures whether fidelity
/// actually improved.
pub fn run_self_correction_loop() -> CorrectionResult {
    use crate::parser::parse_phi_program;
    use crate::phi_ir::evaluator::Evaluator;
    use crate::phi_ir::lowering::lower_program_checked;
    use crate::phi_ir::optimizer::{OptimizationLevel, Optimizer};

    // 1. Initial measurement — decoherent state
    let counts = poll_ibm_job_mock(MockMode::Decoherent);
    let initial_coherence = calculate_coherence(&counts);

    // 2. Check if correction is needed
    let correction_source = generate_correction_if_needed(initial_coherence);

    if correction_source.is_none() {
        return CorrectionResult {
            initial_coherence,
            final_coherence: initial_coherence,
            correction_source: None,
            correction_executed: false,
        };
    }

    let source = correction_source.unwrap();

    // 3. Execute the correction through the Evaluator
    let mut correction_executed = false;
    if let Ok(exprs) = parse_phi_program(&source) {
        if let Ok(mut prog) = lower_program_checked(&exprs) {
            let mut opt = Optimizer::new(OptimizationLevel::Basic);
            opt.optimize(&mut prog);
            let mut eval = Evaluator::new(prog);
            if eval.run().is_ok() {
                correction_executed = true;
            }
        }
    }

    // 4. Re-measure — coherent state after correction
    let new_counts = poll_ibm_job_mock(MockMode::Coherent);
    let final_coherence = calculate_coherence(&new_counts);

    CorrectionResult {
        initial_coherence,
        final_coherence,
        correction_source: Some(source),
        correction_executed,
    }
}
