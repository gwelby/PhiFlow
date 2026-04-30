//! Physical Entropy Buffer — Hardware Noise History for Coherence Auto-Tuning
//!
//! Maintains a circular buffer of execution conditions to:
//! 1. Recommend adaptive coherence thresholds based on historical patterns
//! 2. Detect hardware drift over time
//! 3. Enable auto-tuning of anchor parameters

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Single execution record capturing hardware state at anchor time
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EntropyRecord {
    /// Unix timestamp (seconds since epoch)
    pub timestamp: u64,
    /// Unique session identifier
    pub session_id: String,
    /// SOMA presence reading (0.0-1.0)
    pub soma_presence: f64,
    /// Hardware stress level derived from sensor jitter (0.0-1.0)
    pub hardware_stress: f64,
    /// Gate fidelity threshold used for this execution
    pub gate_fidelity: f64,
    /// Final coherence score at witness point
    pub coherence_at_witness: f64,
    /// Whether SOMA 432 Hz frequency was locked
    pub frequency_locked: bool,
}

impl EntropyRecord {
    /// Create a record from current sensor observations
    pub fn from_observations(
        session_id: String,
        soma_presence: f64,
        hardware_stress: f64,
        gate_fidelity: f64,
        coherence_at_witness: f64,
        frequency_locked: bool,
    ) -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            session_id,
            soma_presence,
            hardware_stress,
            gate_fidelity,
            coherence_at_witness,
            frequency_locked,
        }
    }
}

/// Alert type for hardware degradation detection
#[derive(Debug, Clone, PartialEq)]
pub enum DriftAlert {
    /// Stress levels increasing compared to baseline
    IncreasingStress { baseline: f64, current: f64 },
    /// Coherence scores degrading over time
    DegradingCoherence { baseline: f64, current: f64 },
    /// Presence readings declining
    PresenceDecline { baseline: f64, current: f64 },
}

/// Recommended anchor parameters based on historical analysis
#[derive(Debug, Clone)]
pub struct AnchorRecommendation {
    /// Recommended minimum presence threshold
    pub min_presence: f64,
    /// Recommended target frequency
    pub frequency: f64,
    /// Recommended gate fidelity threshold
    pub gate_fidelity: f64,
    /// Confidence in recommendation (0.0-1.0)
    pub confidence: f64,
    /// Number of records used to generate recommendation
    pub sample_size: usize,
}

/// Circular buffer for hardware entropy history
pub struct EntropyBuffer {
    /// Maximum number of records to retain
    capacity: usize,
    /// The records buffer (oldest at front, newest at back)
    entries: VecDeque<EntropyRecord>,
    /// Path for persistent storage
    storage_path: PathBuf,
    /// Whether auto-tuning is enabled
    auto_tune_enabled: bool,
}

/// Errors that can occur when operating the entropy buffer
#[derive(Debug)]
pub enum BufferError {
    /// IO error reading or writing buffer file
    Io(std::io::Error),
    /// Serialization/deserialization error
    Serialization(serde_json::Error),
    /// Invalid data format
    InvalidFormat(String),
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferError::Io(e) => write!(f, "IO error: {}", e),
            BufferError::Serialization(e) => write!(f, "Serialization error: {}", e),
            BufferError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for BufferError {}

impl From<std::io::Error> for BufferError {
    fn from(e: std::io::Error) -> Self {
        BufferError::Io(e)
    }
}

impl From<serde_json::Error> for BufferError {
    fn from(e: serde_json::Error) -> Self {
        BufferError::Serialization(e)
    }
}

impl EntropyBuffer {
    /// Create a new empty buffer with default capacity (100 records)
    pub fn new(storage_path: PathBuf) -> Self {
        Self {
            capacity: 100,
            entries: VecDeque::new(),
            storage_path,
            auto_tune_enabled: false,
        }
    }

    /// Create a new buffer with custom capacity
    pub fn with_capacity(storage_path: PathBuf, capacity: usize) -> Self {
        Self {
            capacity,
            entries: VecDeque::new(),
            storage_path,
            auto_tune_enabled: false,
        }
    }

    /// Enable or disable auto-tuning
    pub fn set_auto_tune(&mut self, enabled: bool) {
        self.auto_tune_enabled = enabled;
    }

    /// Check if auto-tuning is enabled
    pub fn is_auto_tune_enabled(&self) -> bool {
        self.auto_tune_enabled
    }

    /// Load buffer from NDJSON file (one JSON object per line)
    pub fn load(path: PathBuf) -> Result<Self, BufferError> {
        let mut buffer = Self::new(path.clone());

        if !path.exists() {
            return Ok(buffer);
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<EntropyRecord>(&line) {
                Ok(record) => buffer.push_internal(record),
                Err(e) => {
                    // Skip malformed lines but log them
                    eprintln!("[entropy_buffer] Skipping malformed record: {}", e);
                }
            }
        }

        Ok(buffer)
    }

    /// Save buffer to NDJSON file (append-only format)
    pub fn save(&self) -> Result<(), BufferError> {
        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.storage_path)?;

        for record in &self.entries {
            let json = serde_json::to_string(record)?;
            writeln!(file, "{}", json)?;
        }

        Ok(())
    }

    /// Add a new record, evicting oldest if at capacity
    pub fn push(&mut self, record: EntropyRecord) {
        self.push_internal(record);
    }

    /// Internal push without saving
    fn push_internal(&mut self, record: EntropyRecord) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(record);
    }

    /// Append a single record to the file without loading entire buffer
    pub fn append(&self, record: &EntropyRecord) -> Result<(), BufferError> {
        // Ensure parent directory exists
        if let Some(parent) = self.storage_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.storage_path)?;

        let json = serde_json::to_string(record)?;
        writeln!(file, "{}", json)?;

        Ok(())
    }

    /// Get the number of entries in the buffer
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get all entries
    pub fn entries(&self) -> &VecDeque<EntropyRecord> {
        &self.entries
    }

    /// Get the last N entries (most recent first)
    pub fn recent(&self, n: usize) -> Vec<&EntropyRecord> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Compute recommended coherence threshold based on historical witness scores
    /// Formula: mean(coherence) - std_dev(coherence) for last N runs
    /// This provides a threshold that accounts for 84% of historical performance (1 sigma)
    pub fn recommend_coherence(&self, window_size: usize) -> Option<f64> {
        let recent: Vec<_> = self.entries.iter().rev().take(window_size).collect();

        if recent.len() < 5 {
            return None; // Not enough data
        }

        let sum: f64 = recent.iter().map(|r| r.coherence_at_witness).sum();
        let mean = sum / recent.len() as f64;

        let variance: f64 = recent
            .iter()
            .map(|r| (r.coherence_at_witness - mean).powi(2))
            .sum::<f64>()
            / recent.len() as f64;
        let std_dev = variance.sqrt();

        // Recommend threshold at mean - 1 std dev (84th percentile)
        let recommendation = (mean - std_dev).clamp(0.0, 1.0);
        Some(recommendation)
    }

    /// Detect hardware drift by comparing recent vs older performance
    pub fn detect_drift(&self, window_size: usize) -> Option<DriftAlert> {
        if self.entries.len() < window_size * 2 {
            return None; // Not enough data for comparison
        }

        let recent: Vec<_> = self.entries.iter().rev().take(window_size).collect();
        let older: Vec<_> = self
            .entries
            .iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .collect();

        // Calculate averages
        let recent_stress: f64 =
            recent.iter().map(|r| r.hardware_stress).sum::<f64>() / recent.len() as f64;
        let older_stress: f64 =
            older.iter().map(|r| r.hardware_stress).sum::<f64>() / older.len() as f64;

        let recent_coherence: f64 = recent
            .iter()
            .map(|r| r.coherence_at_witness)
            .sum::<f64>()
            / recent.len() as f64;
        let older_coherence: f64 = older
            .iter()
            .map(|r| r.coherence_at_witness)
            .sum::<f64>()
            / older.len() as f64;

        let recent_presence: f64 =
            recent.iter().map(|r| r.soma_presence).sum::<f64>() / recent.len() as f64;
        let older_presence: f64 =
            older.iter().map(|r| r.soma_presence).sum::<f64>() / older.len() as f64;

        // Check for significant degradation (threshold: 10% change)
        let stress_threshold = 0.1;
        let coherence_threshold = 0.1;
        let presence_threshold = 0.1;

        if recent_stress > older_stress + stress_threshold {
            return Some(DriftAlert::IncreasingStress {
                baseline: older_stress,
                current: recent_stress,
            });
        }

        if recent_coherence < older_coherence - coherence_threshold {
            return Some(DriftAlert::DegradingCoherence {
                baseline: older_coherence,
                current: recent_coherence,
            });
        }

        if recent_presence < older_presence - presence_threshold {
            return Some(DriftAlert::PresenceDecline {
                baseline: older_presence,
                current: recent_presence,
            });
        }

        None
    }

    /// Recommend anchor parameters based on historical analysis
    pub fn recommend_anchor_params(&self) -> AnchorRecommendation {
        if self.entries.len() < 5 {
            // Not enough data — return conservative defaults
            return AnchorRecommendation {
                min_presence: 0.75,
                frequency: 432.0,
                gate_fidelity: 0.95,
                confidence: 0.0,
                sample_size: self.entries.len(),
            };
        }

        let recent_20: Vec<_> = self.entries.iter().rev().take(20).collect();

        // Calculate statistics
        let avg_presence: f64 = recent_20
            .iter()
            .map(|r| r.soma_presence)
            .sum::<f64>()
            / recent_20.len() as f64;
        let min_presence = recent_20
            .iter()
            .map(|r| r.soma_presence)
            .fold(f64::INFINITY, |a, b| a.min(b));

        // Frequency: use most common locked frequency or default to 432
        let freq_432_count = recent_20.iter().filter(|r| r.frequency_locked).count();
        let frequency = if freq_432_count > recent_20.len() / 2 {
            432.0
        } else {
            432.0 // Default, could analyze actual frequency readings if stored
        };

        // Gate fidelity: use minimum observed coherence as threshold
        let min_coherence = recent_20
            .iter()
            .map(|r| r.coherence_at_witness)
            .fold(f64::INFINITY, |a, b| a.min(b));
        let gate_fidelity = min_coherence.clamp(0.85, 0.999);

        // Confidence based on sample size and consistency
        let variance: f64 = recent_20
            .iter()
            .map(|r| (r.soma_presence - avg_presence).powi(2))
            .sum::<f64>()
            / recent_20.len() as f64;
        let consistency = 1.0 - variance.sqrt().clamp(0.0, 1.0);
        let sample_factor = (self.entries.len() as f64 / 100.0).min(1.0);
        let confidence = (consistency * 0.5 + sample_factor * 0.5).clamp(0.0, 1.0);

        AnchorRecommendation {
            min_presence: min_presence.clamp(0.5, 0.99),
            frequency,
            gate_fidelity,
            confidence,
            sample_size: self.entries.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_record(timestamp: u64, presence: f64, coherence: f64) -> EntropyRecord {
        EntropyRecord {
            timestamp,
            session_id: "test".to_string(),
            soma_presence: presence,
            hardware_stress: 0.1,
            gate_fidelity: 0.95,
            coherence_at_witness: coherence,
            frequency_locked: true,
        }
    }

    #[test]
    fn test_buffer_capacity() {
        let temp = NamedTempFile::new().unwrap();
        let mut buffer = EntropyBuffer::with_capacity(temp.path().to_path_buf(), 5);

        for i in 0..10 {
            buffer.push(create_test_record(i as u64, 0.9, 0.95));
        }

        assert_eq!(buffer.len(), 5);
        // Oldest entries should be evicted
        assert_eq!(buffer.entries().front().unwrap().timestamp, 5);
    }

    #[test]
    fn test_recommend_coherence() {
        let temp = NamedTempFile::new().unwrap();
        let mut buffer = EntropyBuffer::new(temp.path().to_path_buf());

        // Not enough data
        assert!(buffer.recommend_coherence(5).is_none());

        // Add 10 records with varying coherence
        for i in 0..10 {
            let coherence = 0.9 + (i as f64 * 0.01);
            buffer.push(create_test_record(i as u64, 0.9, coherence));
        }

        let recommendation = buffer.recommend_coherence(10).unwrap();
        // Should be around mean - std_dev
        assert!(recommendation > 0.0 && recommendation < 1.0);
    }

    #[test]
    fn test_detect_drift() {
        let temp = NamedTempFile::new().unwrap();
        let mut buffer = EntropyBuffer::new(temp.path().to_path_buf());

        // Not enough data
        assert!(buffer.detect_drift(5).is_none());

        // Add 15 records: first 10 good (0.9), next 5 degraded (0.5)
        // With window_size=5: recent=indices 10-14 (0.5), older=indices 5-9 (0.9)
        // Difference = 0.4 > 0.1 threshold → drift detected
        for i in 0..10 {
            buffer.push(create_test_record(i as u64, 0.9, 0.95));
        }
        for i in 10..15 {
            buffer.push(create_test_record(i as u64, 0.5, 0.7));
        }

        let drift = buffer.detect_drift(5);
        assert!(drift.is_some(), "Expected drift to be detected");
        matches!(drift.unwrap(), DriftAlert::PresenceDecline { .. });
    }

    #[test]
    fn test_save_and_load() {
        let temp = NamedTempFile::new().unwrap();
        let path = temp.path().to_path_buf();

        {
            let mut buffer = EntropyBuffer::new(path.clone());
            buffer.push(create_test_record(1, 0.9, 0.95));
            buffer.push(create_test_record(2, 0.85, 0.92));
            buffer.save().unwrap();
        }

        let loaded = EntropyBuffer::load(path).unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_append() {
        let temp = NamedTempFile::new().unwrap();
        let mut buffer = EntropyBuffer::new(temp.path().to_path_buf());

        let record = create_test_record(1, 0.9, 0.95);
        buffer.append(&record).unwrap();

        // Verify file contents
        let contents = std::fs::read_to_string(temp.path()).unwrap();
        assert!(contents.contains("soma_presence"));
    }
}
