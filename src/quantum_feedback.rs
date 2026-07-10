use anyhow::Result;
use std::collections::HashMap;

const PHI_INV: f64 = 0.618033988749895;

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
        let mut mock_counts = HashMap::new();
        mock_counts.insert("00".to_string(), 512);
        mock_counts.insert("11".to_string(), 488); // High coherence (entangled)
        return Ok(mock_counts);
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
/// If physical coherence is lower, generates a self-correcting PhiFlow string.
pub fn generate_correction_if_needed(coherence: f64) -> Option<String> {
    if coherence < PHI_INV {
        // Healing logic: if coherence drops, we evolve a sleep or resonance to stabilize
        let correction = format!(
            r#"
            intention "self_correction" {{
                let low_coherence = {}
                resonate low_coherence
                // Sleep or yield to restore stability
                witness
            }}
            "#,
            coherence
        );
        Some(correction)
    } else {
        None
    }
}
