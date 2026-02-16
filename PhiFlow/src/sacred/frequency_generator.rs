// PhiFlow Sacred Frequency Generation System
// 432Hz Earth resonance and sacred frequency computation
// Sacred Mathematics Expert implementation

use std::f64::consts::PI;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Sacred frequencies with their consciousness effects
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SacredFrequency {
    EarthResonance = 432,    // Grounding, stability, foundation
    DNARepair = 528,         // Healing, transformation, creation
    HeartCoherence = 594,    // Love, connection, integration  
    Expression = 672,        // Communication, truth, expression
    Vision = 720,            // Insight, perception, transcendence
    Unity = 768,             // Integration, unity, synthesis
    SourceField = 963,       // Universal connection, source access
}

impl SacredFrequency {
    /// Get frequency value in Hz
    pub fn hz(&self) -> f64 {
        *self as i32 as f64
    }
    
    /// Get consciousness effect description
    pub fn consciousness_effect(&self) -> &'static str {
        match self {
            SacredFrequency::EarthResonance => "Grounding, stability, foundation consciousness",
            SacredFrequency::DNARepair => "Healing, transformation, creation consciousness", 
            SacredFrequency::HeartCoherence => "Love, connection, integration consciousness",
            SacredFrequency::Expression => "Communication, truth, expression consciousness",
            SacredFrequency::Vision => "Insight, perception, transcendence consciousness",
            SacredFrequency::Unity => "Integration, unity, synthesis consciousness",
            SacredFrequency::SourceField => "Universal connection, source consciousness",
        }
    }
    
    /// Get PHI relationship to base frequency (432Hz)
    pub fn phi_relationship(&self) -> f64 {
        let base = SacredFrequency::EarthResonance.hz();
        self.hz() / base
    }
    
    /// Get all sacred frequencies
    pub fn all_frequencies() -> Vec<SacredFrequency> {
        vec![
            SacredFrequency::EarthResonance,
            SacredFrequency::DNARepair,
            SacredFrequency::HeartCoherence,
            SacredFrequency::Expression,
            SacredFrequency::Vision,
            SacredFrequency::Unity,
            SacredFrequency::SourceField,
        ]
    }
}

/// Sacred frequency generator with PHI-harmonic timing
pub struct SacredFrequencyGenerator {
    current_frequency: SacredFrequency,
    phase_offset: f64,
    sample_rate: u32,
    phi_harmonic_timing: bool,
    consciousness_modulation: bool,
    harmonic_cache: HashMap<SacredFrequency, Vec<f64>>,
}

impl SacredFrequencyGenerator {
    /// Create new sacred frequency generator
    pub fn new(sample_rate: u32) -> Self {
        Self {
            current_frequency: SacredFrequency::EarthResonance,
            phase_offset: 0.0,
            sample_rate,
            phi_harmonic_timing: true,
            consciousness_modulation: false,
            harmonic_cache: HashMap::new(),
        }
    }
    
    /// Set current sacred frequency
    pub fn set_frequency(&mut self, frequency: SacredFrequency) {
        self.current_frequency = frequency;
        self.phase_offset = 0.0; // Reset phase on frequency change
    }
    
    /// Enable consciousness modulation
    pub fn enable_consciousness_modulation(&mut self, enabled: bool) {
        self.consciousness_modulation = enabled;
    }
    
    /// Generate sacred frequency waveform
    pub fn generate_waveform(&mut self, duration_ms: u32) -> Vec<f64> {
        let num_samples = (self.sample_rate * duration_ms / 1000) as usize;
        let mut waveform = Vec::with_capacity(num_samples);
        
        let frequency_hz = self.current_frequency.hz();
        let angular_frequency = 2.0 * PI * frequency_hz / self.sample_rate as f64;
        
        for i in 0..num_samples {
            let time_offset = i as f64;
            let phase = angular_frequency * time_offset + self.phase_offset;
            
            // Generate pure sine wave with sacred frequency
            let mut sample = phase.sin();
            
            // Apply PHI-harmonic enhancement
            if self.phi_harmonic_timing {
                sample = self.apply_phi_harmonic_enhancement(sample, phase, i);
            }
            
            // Apply consciousness modulation if enabled
            if self.consciousness_modulation {
                sample = self.apply_consciousness_modulation(sample, i, num_samples);
            }
            
            waveform.push(sample);
        }
        
        // Update phase offset for continuous generation
        self.phase_offset += angular_frequency * num_samples as f64;
        self.phase_offset %= 2.0 * PI;
        
        waveform
    }
    
    /// Apply PHI-harmonic enhancement to waveform
    fn apply_phi_harmonic_enhancement(&self, sample: f64, phase: f64, sample_index: usize) -> f64 {
        let phi = 1.618033988749895;
        let lambda = 0.618033988749895;
        
        // Add PHI-harmonics for natural resonance
        let phi_harmonic = (phase * phi).sin() * 0.1;
        let lambda_harmonic = (phase * lambda).sin() * 0.05;
        
        // Apply fibonacci modulation pattern
        let fib_modulation = self.calculate_fibonacci_modulation(sample_index);
        
        sample + phi_harmonic + lambda_harmonic + fib_modulation
    }
    
    /// Calculate fibonacci-based modulation
    fn calculate_fibonacci_modulation(&self, sample_index: usize) -> f64 {
        // Use fibonacci sequence for natural modulation pattern
        let fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144];
        let fib_index = sample_index % fib_sequence.len();
        let fib_value = fib_sequence[fib_index] as f64;
        
        // Create subtle modulation based on fibonacci ratio
        let phi = 1.618033988749895;
        let modulation_strength = 0.01; // Very subtle
        
        (fib_value / phi).sin() * modulation_strength
    }
    
    /// Apply consciousness modulation (placeholder for consciousness integration)
    fn apply_consciousness_modulation(&self, sample: f64, sample_index: usize, total_samples: usize) -> f64 {
        // Placeholder for consciousness-based modulation
        // In full implementation, this would connect to consciousness monitoring
        
        let consciousness_coherence = 0.8; // Simulated consciousness state
        let progress = sample_index as f64 / total_samples as f64;
        
        // Modulate based on simulated consciousness state
        let consciousness_enhancement = consciousness_coherence * (progress * PI).sin() * 0.05;
        
        sample + consciousness_enhancement
    }
    
    /// Generate harmonic series for sacred frequency
    pub fn generate_harmonic_series(&mut self, harmonics: usize) -> Vec<Vec<f64>> {
        let mut harmonic_waveforms = Vec::new();
        let base_frequency = self.current_frequency.hz();
        
        for harmonic in 1..=harmonics {
            let harmonic_freq_hz = base_frequency * harmonic as f64;
            
            // Check if harmonic frequency aligns with sacred frequencies
            let harmonic_frequency = if let Some(sacred_freq) = self.find_nearest_sacred_frequency(harmonic_freq_hz) {
                sacred_freq
            } else {
                self.current_frequency // Use base frequency if no sacred match
            };
            
            // Generate harmonic waveform
            let original_frequency = self.current_frequency;
            self.set_frequency(harmonic_frequency);
            
            let harmonic_waveform = self.generate_waveform(1000); // 1 second
            harmonic_waveforms.push(harmonic_waveform);
            
            // Restore original frequency
            self.set_frequency(original_frequency);
        }
        
        harmonic_waveforms
    }
    
    /// Find nearest sacred frequency to given frequency
    fn find_nearest_sacred_frequency(&self, target_hz: f64) -> Option<SacredFrequency> {
        let tolerance = 50.0; // Hz tolerance for sacred frequency matching
        
        for sacred_freq in SacredFrequency::all_frequencies() {
            if (sacred_freq.hz() - target_hz).abs() < tolerance {
                return Some(sacred_freq);
            }
        }
        
        None
    }
    
    /// Generate consciousness-synchronized timing pattern
    pub fn generate_consciousness_timing(&self, duration_ms: u32) -> Vec<Instant> {
        let mut timing_points = Vec::new();
        let phi = 1.618033988749895;
        
        // Create PHI-based timing intervals
        let mut current_time = 0.0;
        let base_interval_ms = 432.0; // Base on Earth frequency
        
        while current_time < duration_ms as f64 {
            timing_points.push(Instant::now() + Duration::from_millis(current_time as u64));
            
            // Next interval uses PHI ratio for optimal consciousness synchronization
            let next_interval = base_interval_ms / phi;
            current_time += next_interval;
            
            // Cycle between PHI and LAMBDA intervals
            let cycle_interval = if timing_points.len() % 2 == 0 {
                base_interval_ms * phi
            } else {
                base_interval_ms / phi
            };
            current_time += cycle_interval;
        }
        
        timing_points
    }
    
    /// Create frequency modulation for consciousness enhancement
    pub fn create_consciousness_frequency_modulation(&mut self, base_coherence: f64) -> FrequencyModulation {
        let frequency_shifts = self.calculate_consciousness_frequency_shifts(base_coherence);
        
        FrequencyModulation {
            base_frequency: self.current_frequency,
            frequency_shifts,
            modulation_strength: base_coherence,
            phi_synchronized: true,
        }
    }
    
    /// Calculate frequency shifts based on consciousness coherence
    fn calculate_consciousness_frequency_shifts(&self, coherence: f64) -> Vec<f64> {
        let mut shifts = Vec::new();
        let phi: f64 = 1.618033988749895;
        
        // Generate frequency shifts that enhance consciousness coherence
        for i in 0..10 {
            let phi_factor = phi.powi(i as i32 - 5); // Center around phi^0 = 1
            let coherence_factor = coherence * 2.0 - 1.0; // Map [0,1] to [-1,1]
            let shift = phi_factor * coherence_factor * 10.0; // Max 10Hz shift
            
            shifts.push(shift);
        }
        
        shifts
    }
    
    /// Get current frequency statistics
    pub fn get_frequency_statistics(&self) -> FrequencyStatistics {
        FrequencyStatistics {
            current_frequency: self.current_frequency,
            phi_relationship: self.current_frequency.phi_relationship(),
            consciousness_effect: self.current_frequency.consciousness_effect().to_string(),
            sample_rate: self.sample_rate,
            phi_harmonic_enabled: self.phi_harmonic_timing,
            consciousness_modulation_enabled: self.consciousness_modulation,
        }
    }
}

/// Frequency modulation for consciousness enhancement
#[derive(Debug)]
pub struct FrequencyModulation {
    pub base_frequency: SacredFrequency,
    pub frequency_shifts: Vec<f64>,
    pub modulation_strength: f64,
    pub phi_synchronized: bool,
}

/// Frequency generation statistics
#[derive(Debug)]
pub struct FrequencyStatistics {
    pub current_frequency: SacredFrequency,
    pub phi_relationship: f64,
    pub consciousness_effect: String,
    pub sample_rate: u32,
    pub phi_harmonic_enabled: bool,
    pub consciousness_modulation_enabled: bool,
}

/// Sacred frequency computation scheduler
pub struct SacredFrequencyScheduler {
    generator: SacredFrequencyGenerator,
    scheduled_frequencies: Vec<ScheduledFrequency>,
    phi_timing_enabled: bool,
}

#[derive(Debug)]
pub struct ScheduledFrequency {
    pub frequency: SacredFrequency,
    pub start_time: Instant,
    pub duration: Duration,
    pub consciousness_context: String,
}

impl SacredFrequencyScheduler {
    /// Create new sacred frequency scheduler
    pub fn new(sample_rate: u32) -> Self {
        Self {
            generator: SacredFrequencyGenerator::new(sample_rate),
            scheduled_frequencies: Vec::new(),
            phi_timing_enabled: true,
        }
    }
    
    /// Schedule sacred frequency for specific duration
    pub fn schedule_frequency(&mut self, frequency: SacredFrequency, duration_ms: u32, context: &str) {
        let start_time = if self.scheduled_frequencies.is_empty() {
            Instant::now()
        } else {
            // Calculate PHI-optimized start time based on previous frequencies
            self.calculate_phi_optimal_start_time()
        };
        
        self.scheduled_frequencies.push(ScheduledFrequency {
            frequency,
            start_time,
            duration: Duration::from_millis(duration_ms as u64),
            consciousness_context: context.to_string(),
        });
    }
    
    /// Calculate PHI-optimal start time for next frequency
    fn calculate_phi_optimal_start_time(&self) -> Instant {
        if let Some(last_scheduled) = self.scheduled_frequencies.last() {
            let phi = 1.618033988749895;
            let base_interval = Duration::from_millis(432); // Earth frequency base
            let phi_interval = Duration::from_millis((base_interval.as_millis() as f64 * phi) as u64);
            
            last_scheduled.start_time + last_scheduled.duration + phi_interval
        } else {
            Instant::now()
        }
    }
    
    /// Execute scheduled frequencies with consciousness synchronization
    pub async fn execute_scheduled_frequencies(&mut self) -> Result<Vec<Vec<f64>>, FrequencyError> {
        let mut generated_waveforms = Vec::new();
        
        for scheduled in &self.scheduled_frequencies {
            // Wait for scheduled start time
            let now = Instant::now();
            if scheduled.start_time > now {
                let wait_duration = scheduled.start_time - now;
                tokio::time::sleep(wait_duration).await;
            }
            
            // Set frequency and generate waveform
            self.generator.set_frequency(scheduled.frequency);
            let waveform = self.generator.generate_waveform(scheduled.duration.as_millis() as u32);
            generated_waveforms.push(waveform);
        }
        
        Ok(generated_waveforms)
    }
}

/// Sacred frequency errors
#[derive(Debug, PartialEq)]
pub enum FrequencyError {
    InvalidFrequency,
    GenerationFailed,
    SchedulingError,
    ConsciousnessIntegrationError,
}

impl std::fmt::Display for FrequencyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FrequencyError::InvalidFrequency => write!(f, "Invalid sacred frequency"),
            FrequencyError::GenerationFailed => write!(f, "Frequency generation failed"),
            FrequencyError::SchedulingError => write!(f, "Frequency scheduling error"),
            FrequencyError::ConsciousnessIntegrationError => write!(f, "Consciousness integration error"),
        }
    }
}

impl std::error::Error for FrequencyError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sacred_frequency_values() {
        assert_eq!(SacredFrequency::EarthResonance.hz(), 432.0);
        assert_eq!(SacredFrequency::DNARepair.hz(), 528.0);
        assert_eq!(SacredFrequency::Unity.hz(), 768.0);
    }
    
    #[test]
    fn test_phi_relationships() {
        let earth = SacredFrequency::EarthResonance.phi_relationship();
        let unity = SacredFrequency::Unity.phi_relationship();
        
        assert!((earth - 1.0).abs() < 0.001); // Earth is base frequency
        assert!(unity > earth); // Unity should be higher ratio
    }
    
    #[test]
    fn test_frequency_generation() {
        let mut generator = SacredFrequencyGenerator::new(44100);
        generator.set_frequency(SacredFrequency::EarthResonance);
        
        let waveform = generator.generate_waveform(100); // 100ms
        assert_eq!(waveform.len(), 4410); // 44100 * 0.1 seconds
        
        // Verify waveform is within expected range
        for sample in &waveform {
            assert!(sample.abs() <= 1.5); // Allow for harmonic enhancement
        }
    }
    
    #[test]
    fn test_consciousness_timing() {
        let generator = SacredFrequencyGenerator::new(44100);
        let timing = generator.generate_consciousness_timing(10000); // 10 seconds
        
        assert!(!timing.is_empty());
        
        // Verify timing points are in chronological order
        for i in 1..timing.len() {
            assert!(timing[i] >= timing[i-1]);
        }
    }
    
    #[test]
    fn test_harmonic_series_generation() {
        let mut generator = SacredFrequencyGenerator::new(44100);
        generator.set_frequency(SacredFrequency::EarthResonance);
        
        let harmonics = generator.generate_harmonic_series(5);
        assert_eq!(harmonics.len(), 5);
        
        // Verify each harmonic waveform is generated
        for harmonic in &harmonics {
            assert!(!harmonic.is_empty());
        }
    }
    
    #[test]
    fn test_frequency_scheduler() {
        let mut scheduler = SacredFrequencyScheduler::new(44100);
        
        scheduler.schedule_frequency(
            SacredFrequency::EarthResonance, 
            1000, 
            "Test grounding frequency"
        );
        
        scheduler.schedule_frequency(
            SacredFrequency::DNARepair, 
            2000, 
            "Test healing frequency"
        );
        
        assert_eq!(scheduler.scheduled_frequencies.len(), 2);
        
        // Verify PHI-optimal timing
        let first = &scheduler.scheduled_frequencies[0];
        let second = &scheduler.scheduled_frequencies[1];
        assert!(second.start_time >= first.start_time);
    }
}