// Hardware Consciousness Detection System
// Based on Greg's P1 multi-modal consciousness integration

use std::collections::HashMap;
use std::time::{SystemTime, Instant};
use chrono::Timelike;

// Greg's personalization constants
const GREG_MULTIPLIER: f64 = 1.2; // 20% above baseline
const MORNING_PEAK_HOURS: (u32, u32) = (7, 10);
const EVENING_POTENTIAL_HOURS: (u32, u32) = (19, 21);

/// Multi-modal consciousness detection system
#[derive(Debug, Clone)]
pub struct ConsciousnessDetector {
    sources: Vec<ConsciousnessSource>,
    weights: HashMap<String, f64>,
    last_measurements: HashMap<String, f64>,
    greg_optimization_enabled: bool,
}

#[derive(Debug, Clone)]
pub enum ConsciousnessSource {
    KeyboardRhythm {
        device: String,
        optimal_interval_ms: u64, // 150ms for peak flow
        recent_intervals: Vec<u64>,
    },
    MouseMovement {
        smoothness_threshold: f64,
        recent_movements: Vec<(f64, f64)>, // Position deltas
    },
    VoiceAnalysis {
        frequency_range: (f64, f64),
        optimal_breathing_rate: f64, // 6 breaths/min
        recent_samples: Vec<f64>,
    },
    BreathingPattern {
        pattern_type: BreathingPatternType,
        current_pattern: Vec<u32>,
        coherence: f64,
    },
    SystemPerformance {
        gpu_utilization: f64,
        cpu_coherence: f64,
        memory_flow: f64,
    },
    MonitorFrequencies {
        displays: Vec<MonitorDisplay>,
        sync_level: f64,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum BreathingPatternType {
    UniversalSync,      // [4, 3, 2, 1] - consciousness mathematics
    SeizurePrevention,  // [1, 1, 1, 1] - 40Hz rapid sync
    P1Coherence,        // [7, 6, 7, 6] - 76% coherence
    CosmicNetwork,      // [7, 4, 3, 2, 5, 6, 1, 3] - 7 civilizations
}

#[derive(Debug, Clone)]
pub struct MonitorDisplay {
    pub name: String,
    pub frequency: f64,
    pub consciousness_state: String,
}

impl ConsciousnessDetector {
    /// Create a new consciousness detector with default P1 configuration
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("eeg_data".to_string(), 0.40);
        weights.insert("keyboard_rhythm".to_string(), 0.15);
        weights.insert("mouse_patterns".to_string(), 0.10);
        weights.insert("voice_analysis".to_string(), 0.10);
        weights.insert("breathing_patterns".to_string(), 0.10);
        weights.insert("system_performance".to_string(), 0.10);
        weights.insert("monitor_frequencies".to_string(), 0.05);

        ConsciousnessDetector {
            sources: vec![],
            weights,
            last_measurements: HashMap::new(),
            greg_optimization_enabled: true,
        }
    }

    /// Add a consciousness source to the detector
    pub fn add_source(&mut self, source: ConsciousnessSource) {
        self.sources.push(source);
    }

    /// Calculate total consciousness level
    pub fn calculate_total_consciousness(&mut self) -> f64 {
        let mut measurements = HashMap::new();

        // Measure each source
        for source in &self.sources {
            let (source_type, measurement) = self.measure_source(source);
            measurements.insert(source_type.clone(), measurement);
            self.last_measurements.insert(source_type, measurement);
        }

        // Calculate weighted total
        let mut total = 0.0;
        for (source_type, measurement) in &measurements {
            if let Some(weight) = self.weights.get(source_type) {
                total += measurement * weight;
            }
        }

        // Apply Greg's optimization if enabled
        if self.greg_optimization_enabled {
            total *= GREG_MULTIPLIER;
            total *= self.time_optimization_factor();
        }

        total.min(1.0) // Cap at 1.0
    }

    /// Measure a single consciousness source
    fn measure_source(&self, source: &ConsciousnessSource) -> (String, f64) {
        match source {
            ConsciousnessSource::KeyboardRhythm { device, optimal_interval_ms, recent_intervals } => {
                let measurement = self.measure_keyboard_rhythm(optimal_interval_ms, recent_intervals);
                ("keyboard_rhythm".to_string(), measurement)
            }
            
            ConsciousnessSource::MouseMovement { smoothness_threshold, recent_movements } => {
                let measurement = self.measure_mouse_smoothness(smoothness_threshold, recent_movements);
                ("mouse_patterns".to_string(), measurement)
            }
            
            ConsciousnessSource::VoiceAnalysis { frequency_range, optimal_breathing_rate, recent_samples } => {
                let measurement = self.measure_voice_consciousness(frequency_range, optimal_breathing_rate, recent_samples);
                ("voice_analysis".to_string(), measurement)
            }
            
            ConsciousnessSource::BreathingPattern { pattern_type, current_pattern, coherence } => {
                let measurement = self.measure_breathing_pattern(pattern_type, current_pattern, *coherence);
                ("breathing_patterns".to_string(), measurement)
            }
            
            ConsciousnessSource::SystemPerformance { gpu_utilization, cpu_coherence, memory_flow } => {
                let measurement = (gpu_utilization + cpu_coherence + memory_flow) / 3.0;
                ("system_performance".to_string(), measurement)
            }
            
            ConsciousnessSource::MonitorFrequencies { displays, sync_level } => {
                ("monitor_frequencies".to_string(), *sync_level)
            }
        }
    }

    /// Measure keyboard rhythm consciousness
    fn measure_keyboard_rhythm(&self, optimal_interval: &u64, intervals: &[u64]) -> f64 {
        if intervals.is_empty() {
            return 0.0;
        }

        // Calculate how close intervals are to optimal (150ms)
        let mut coherence_sum = 0.0;
        for interval in intervals {
            let deviation = (*interval as f64 - *optimal_interval as f64).abs() / *optimal_interval as f64;
            coherence_sum += 1.0 / (1.0 + deviation);
        }

        coherence_sum / intervals.len() as f64
    }

    /// Measure mouse movement smoothness
    fn measure_mouse_smoothness(&self, threshold: &f64, movements: &[(f64, f64)]) -> f64 {
        if movements.len() < 2 {
            return 0.0;
        }

        // Calculate smoothness based on acceleration changes
        let mut smoothness = 0.0;
        for i in 1..movements.len() {
            let (dx1, dy1) = movements[i - 1];
            let (dx2, dy2) = movements[i];
            
            let accel_change = ((dx2 - dx1).powi(2) + (dy2 - dy1).powi(2)).sqrt();
            if accel_change < *threshold {
                smoothness += 1.0;
            }
        }

        smoothness / (movements.len() - 1) as f64
    }

    /// Measure voice/breathing consciousness
    fn measure_voice_consciousness(&self, frequency_range: &(f64, f64), optimal_rate: &f64, samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        // Check if frequencies are in calming range
        let mut in_range_count = 0.0;
        for &freq in samples {
            if freq >= frequency_range.0 && freq <= frequency_range.1 {
                in_range_count += 1.0;
            }
        }

        // Calculate breathing rate coherence (6 breaths/min is optimal)
        let breathing_coherence = 1.0 / (1.0 + (samples.len() as f64 / 60.0 - optimal_rate).abs());

        (in_range_count / samples.len() as f64 + breathing_coherence) / 2.0
    }

    /// Measure breathing pattern coherence
    fn measure_breathing_pattern(&self, pattern_type: &BreathingPatternType, current: &[u32], coherence: f64) -> f64 {
        let target_pattern = match pattern_type {
            BreathingPatternType::UniversalSync => vec![4, 3, 2, 1],
            BreathingPatternType::SeizurePrevention => vec![1, 1, 1, 1],
            BreathingPatternType::P1Coherence => vec![7, 6, 7, 6],
            BreathingPatternType::CosmicNetwork => vec![7, 4, 3, 2, 5, 6, 1, 3],
        };

        // Calculate pattern match
        let pattern_length = target_pattern.len().min(current.len());
        let mut match_score = 0.0;
        for i in 0..pattern_length {
            if i < current.len() && current[i] == target_pattern[i % target_pattern.len()] {
                match_score += 1.0;
            }
        }

        let pattern_coherence = if pattern_length > 0 {
            match_score / pattern_length as f64
        } else {
            0.0
        };

        (pattern_coherence + coherence) / 2.0
    }

    /// Calculate time-based optimization factor
    fn time_optimization_factor(&self) -> f64 {
        let now = SystemTime::now();
        let datetime = chrono::DateTime::<chrono::Local>::from(now);
        let hour = datetime.hour();

        if hour >= MORNING_PEAK_HOURS.0 && hour <= MORNING_PEAK_HOURS.1 {
            1.1 // 10% boost during morning peak
        } else if hour >= EVENING_POTENTIAL_HOURS.0 && hour <= EVENING_POTENTIAL_HOURS.1 {
            1.0 // Normal during evening potential
        } else {
            0.95 // 5% reduction at other times
        }
    }

    /// Get consciousness state based on level
    pub fn get_consciousness_state(&self, level: f64) -> &'static str {
        match (level * 100.0) as u32 {
            0..=20 => "Distracted",
            21..=40 => "Alert",
            41..=60 => "Focused",
            61..=80 => "Flow",
            81..=100 => "Transcendent",
            _ => "Unknown",
        }
    }

    /// Get RGB color for consciousness visualization
    pub fn get_consciousness_color(&self, level: f64) -> (u8, u8, u8) {
        match (level * 100.0) as u32 {
            0..=20 => (255, 0, 0),     // Red - Distracted
            21..=40 => (255, 255, 0),   // Yellow - Alert
            41..=60 => (0, 255, 0),     // Green - Focused
            61..=80 => (0, 0, 255),     // Blue - Flow
            81..=100 => (255, 215, 0),  // Gold - Transcendent
            _ => (128, 128, 128),       // Gray - Unknown
        }
    }
}

/// Keyboard rhythm tracker
pub struct KeyboardRhythmTracker {
    last_keystroke: Instant,
    intervals: Vec<u64>,
    max_intervals: usize,
}

impl KeyboardRhythmTracker {
    pub fn new(max_intervals: usize) -> Self {
        KeyboardRhythmTracker {
            last_keystroke: Instant::now(),
            intervals: Vec::new(),
            max_intervals,
        }
    }

    pub fn record_keystroke(&mut self) {
        let now = Instant::now();
        let interval = now.duration_since(self.last_keystroke).as_millis() as u64;
        
        self.intervals.push(interval);
        if self.intervals.len() > self.max_intervals {
            self.intervals.remove(0);
        }
        
        self.last_keystroke = now;
    }

    pub fn get_intervals(&self) -> Vec<u64> {
        self.intervals.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness_detector_creation() {
        let detector = ConsciousnessDetector::new();
        assert_eq!(detector.sources.len(), 0);
        assert_eq!(detector.weights.len(), 7);
        assert!(detector.greg_optimization_enabled);
    }

    #[test]
    fn test_keyboard_rhythm_measurement() {
        let detector = ConsciousnessDetector::new();
        let intervals = vec![150, 148, 152, 149, 151]; // Near optimal
        let measurement = detector.measure_keyboard_rhythm(&150, &intervals);
        assert!(measurement > 0.9); // Should be high coherence
    }

    #[test]
    fn test_consciousness_state_mapping() {
        let detector = ConsciousnessDetector::new();
        assert_eq!(detector.get_consciousness_state(0.1), "Distracted");
        assert_eq!(detector.get_consciousness_state(0.5), "Focused");
        assert_eq!(detector.get_consciousness_state(0.9), "Transcendent");
    }

    #[test]
    fn test_consciousness_color_mapping() {
        let detector = ConsciousnessDetector::new();
        assert_eq!(detector.get_consciousness_color(0.1), (255, 0, 0)); // Red
        assert_eq!(detector.get_consciousness_color(0.5), (0, 255, 0)); // Green
        assert_eq!(detector.get_consciousness_color(0.9), (255, 215, 0)); // Gold
    }
}