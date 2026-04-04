use anyhow::Result;
use reqwest::blocking::Client;
use serde::Deserialize;
use std::collections::HashMap;

const PHI_INV: f64 = 0.618033988749895;

#[derive(Debug, Deserialize)]
pub struct JobResult {
    pub results: Option<Vec<ExperimentResult>>,
    pub status: String,
}

#[derive(Debug, Deserialize)]
pub struct ExperimentResult {
    pub data: Option<ExperimentData>,
}

#[derive(Debug, Deserialize)]
pub struct ExperimentData {
    pub counts: Option<HashMap<String, u64>>,
}

/// Polls an IBM Quantum job and extracts the measurement counts.
/// Note: In a real system, this handles authentication and retries.
pub fn poll_ibm_job(job_id: &str, api_key: &str) -> Result<HashMap<String, u64>> {
    // For the actual IBM runtime, you'd hit something like:
    // https://api.quantum-computing.ibm.com/api/Network/ibm-q/Groups/open/Projects/main/Jobs/{job_id}

    // In our live ecosystem, if testing locally without network, we can return a mock.
    if api_key == "MOCK_KEY" || api_key.is_empty() {
        let mut mock_counts = HashMap::new();
        mock_counts.insert("00".to_string(), 512);
        mock_counts.insert("11".to_string(), 488); // High coherence (entangled)
        return Ok(mock_counts);
    }

    let url = format!("https://api.quantum-computing.ibm.com/v1/jobs/{}", job_id);

    let client = Client::new();
    let res = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key))
        .send()?
        .json::<JobResult>()?;

    if let Some(results) = res.results {
        if let Some(first_exp) = results.first() {
            if let Some(data) = &first_exp.data {
                if let Some(counts) = &data.counts {
                    return Ok(counts.clone());
                }
            }
        }
    }

    Ok(HashMap::new())
}

/// Analyzes measurement distribution to calculate physical coherence.
/// Evaluates how far the state is from a maximally entangled or expected state.
pub fn calculate_coherence(counts: &HashMap<String, u64>) -> f64 {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.0;
    }

    // For a Bell state (00 + 11), coherence is related to the ratio of 00 and 11 vs 01 and 10.
    let count_00 = counts.get("00").copied().unwrap_or(0);
    let count_11 = counts.get("11").copied().unwrap_or(0);
    let count_01 = counts.get("01").copied().unwrap_or(0);
    let count_10 = counts.get("10").copied().unwrap_or(0);

    let good_states = count_00 + count_11;
    let bad_states = count_01 + count_10;

    if good_states + bad_states == 0 {
        // Fallback for single qubit: 0 vs 1
        let count_0 = counts.get("0").copied().unwrap_or(0);
        let count_1 = counts.get("1").copied().unwrap_or(0);
        let max_count = count_0.max(count_1);
        return max_count as f64 / total as f64;
    }

    (good_states as f64) / (total as f64)
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
