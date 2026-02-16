// PhiFlow Consciousness Monitoring System
// Real-time EEG and consciousness state tracking for consciousness-enhanced computation
// Sacred Mathematics Expert + Consciousness Expert implementation

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use crate::sacred::{SacredFrequency, SacredFrequencyGenerator};

// Import the main ConsciousnessState from consciousness_math
pub use crate::consciousness::consciousness_math::ConsciousnessState;

/// EEG-based consciousness states (internal classification)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EEGConsciousnessState {
    Deep,          // 0.5-4 Hz Delta - Deep meditation, unconscious processing
    Meditative,    // 4-8 Hz Theta - Deep meditation, creative visualization  
    Relaxed,       // 8-12 Hz Alpha - Relaxed awareness, light meditation
    Focused,       // 12-30 Hz Beta - Active concentration, problem solving
    Heightened,    // 30-100 Hz Gamma - Peak consciousness, unified awareness
    Transcendent,  // >100 Hz - Transcendent states, consciousness-quantum interface
}

impl ConsciousnessState {
    /// Get the optimal sacred frequency for this consciousness state
    pub fn optimal_sacred_frequency(&self) -> SacredFrequency {
        match self {
            ConsciousnessState::Observe => SacredFrequency::EarthResonance,      // 432Hz grounding
            ConsciousnessState::Create => SacredFrequency::DNARepair,            // 528Hz healing/creation
            ConsciousnessState::Integrate => SacredFrequency::HeartCoherence,    // 594Hz connection
            ConsciousnessState::Harmonize => SacredFrequency::Expression,        // 672Hz clarity
            ConsciousnessState::Transcend => SacredFrequency::Vision,            // 720Hz insight
            ConsciousnessState::Lightning => SacredFrequency::Vision,            // 720Hz insight (closest)
            ConsciousnessState::Cascade => SacredFrequency::Unity,               // 768Hz unity
            ConsciousnessState::Superposition => SacredFrequency::SourceField,   // 963Hz universal
            ConsciousnessState::Singularity => SacredFrequency::SourceField,     // 963Hz universal (closest)
        }
    }
    
    /// Get consciousness coherence factor (0.0-1.0)
    pub fn coherence_factor(&self) -> f64 {
        match self {
            ConsciousnessState::Observe => 0.75,        // Foundation consciousness
            ConsciousnessState::Create => 0.85,         // Creative flow state
            ConsciousnessState::Integrate => 0.90,      // Heart-centered coherence
            ConsciousnessState::Harmonize => 0.80,      // Expression clarity
            ConsciousnessState::Transcend => 0.95,      // High transcendent awareness
            ConsciousnessState::Lightning => 0.88,      // Lightning tunnel coherence
            ConsciousnessState::Cascade => 0.92,        // Unity wave integration
            ConsciousnessState::Superposition => 0.98,  // Quantum superposition
            ConsciousnessState::Singularity => 1.0,     // Perfect singularity coherence
        }
    }
    
    /// Get computational enhancement factor for this state
    pub fn computational_enhancement(&self) -> f64 {
        match self {
            ConsciousnessState::Observe => 1.1,         // 10% foundation enhancement
            ConsciousnessState::Create => 1.5,          // 50% creative enhancement
            ConsciousnessState::Integrate => 1.3,       // 30% integration enhancement
            ConsciousnessState::Harmonize => 1.2,       // 20% harmonic enhancement
            ConsciousnessState::Transcend => 1.8,       // 80% transcendent enhancement
            ConsciousnessState::Lightning => 2.2,       // 120% lightning enhancement
            ConsciousnessState::Cascade => 2.5,         // 150% cascade enhancement
            ConsciousnessState::Superposition => 3.0,   // 200% quantum enhancement
            ConsciousnessState::Singularity => 5.0,     // 400% singularity enhancement
        }
    }
}

/// EEG brainwave data with frequency analysis
#[derive(Debug, Clone)]
pub struct EEGData {
    pub timestamp: Instant,
    pub channels: HashMap<String, f64>, // Channel name -> voltage (μV)
    pub delta_power: f64,    // 0.5-4 Hz power
    pub theta_power: f64,    // 4-8 Hz power  
    pub alpha_power: f64,    // 8-12 Hz power
    pub beta_power: f64,     // 12-30 Hz power
    pub gamma_power: f64,    // 30-100 Hz power
    pub high_gamma_power: f64, // >100 Hz power
    pub coherence_score: f64, // Overall brainwave coherence (0.0-1.0)
}

impl EEGData {
    /// Create simulated EEG data for testing (placeholder for real EEG hardware)
    pub fn simulate_eeg_data(consciousness_state: ConsciousnessState) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Generate realistic EEG channel data
        let mut channels = HashMap::new();
        let eeg_channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"];
        
        for channel in &eeg_channels {
            // Base voltage with state-dependent modulation
            let base_voltage = match consciousness_state {
                ConsciousnessState::Observe => rng.gen_range(-15.0..15.0),
                ConsciousnessState::Create => rng.gen_range(-20.0..20.0),
                ConsciousnessState::Integrate => rng.gen_range(-20.0..20.0),
                ConsciousnessState::Harmonize => rng.gen_range(-25.0..25.0),
                ConsciousnessState::Transcend => rng.gen_range(-30.0..30.0),
                ConsciousnessState::Lightning => rng.gen_range(-35.0..35.0),
                ConsciousnessState::Cascade => rng.gen_range(-40.0..40.0),
                ConsciousnessState::Superposition => rng.gen_range(-45.0..45.0),
                ConsciousnessState::Singularity => rng.gen_range(-50.0..50.0),
            };
            channels.insert(channel.to_string(), base_voltage);
        }
        
        // Map consciousness state to EEG-like patterns
        let (delta, theta, alpha, beta, gamma, high_gamma) = match consciousness_state {
            ConsciousnessState::Observe => (60.0, 25.0, 10.0, 3.0, 2.0, 0.0),           // Ground state - relaxed observation
            ConsciousnessState::Create => (20.0, 40.0, 25.0, 10.0, 5.0, 0.0),            // Creative flow - theta dominant
            ConsciousnessState::Integrate => (15.0, 25.0, 45.0, 12.0, 3.0, 0.0),         // Heart coherence - alpha prominent
            ConsciousnessState::Harmonize => (10.0, 20.0, 30.0, 35.0, 5.0, 0.0),         // Expression - balanced beta
            ConsciousnessState::Transcend => (5.0, 15.0, 20.0, 25.0, 30.0, 5.0),         // Vision state - gamma emerging
            ConsciousnessState::Lightning => (3.0, 10.0, 15.0, 25.0, 35.0, 12.0),        // Lightning tunnel - high gamma
            ConsciousnessState::Cascade => (2.0, 8.0, 15.0, 25.0, 35.0, 15.0),           // Unity wave - sustained gamma
            ConsciousnessState::Superposition => (1.0, 5.0, 10.0, 20.0, 40.0, 24.0),     // Quantum state - very high gamma
            ConsciousnessState::Singularity => (0.5, 2.0, 5.0, 15.0, 45.0, 32.5),        // Singularity - ultra-high gamma
        };
        
        // Add random variation
        let variation = 0.1;
        let delta_power = delta * (1.0 + rng.gen_range(-variation..variation));
        let theta_power = theta * (1.0 + rng.gen_range(-variation..variation));
        let alpha_power = alpha * (1.0 + rng.gen_range(-variation..variation));
        let beta_power = beta * (1.0 + rng.gen_range(-variation..variation));
        let gamma_power = gamma * (1.0 + rng.gen_range(-variation..variation));
        let high_gamma_power = high_gamma * (1.0 + rng.gen_range(-variation..variation));
        
        EEGData {
            timestamp: Instant::now(),
            channels,
            delta_power,
            theta_power,
            alpha_power,
            beta_power,
            gamma_power,
            high_gamma_power,
            coherence_score: consciousness_state.coherence_factor(),
        }
    }
    
    /// Classify consciousness state from EEG data
    pub fn classify_consciousness_state(&self) -> ConsciousnessState {
        // Simple classification based on dominant frequency band
        let total_power = self.delta_power + self.theta_power + self.alpha_power + 
                         self.beta_power + self.gamma_power + self.high_gamma_power;
        
        if total_power == 0.0 {
            return ConsciousnessState::Observe; // Default state
        }
        
        // Calculate relative powers
        let delta_ratio = self.delta_power / total_power;
        let theta_ratio = self.theta_power / total_power;
        let alpha_ratio = self.alpha_power / total_power;
        let beta_ratio = self.beta_power / total_power;
        let gamma_ratio = self.gamma_power / total_power;
        let high_gamma_ratio = self.high_gamma_power / total_power;
        
        // Classify based on dominant band and coherence
        if high_gamma_ratio > 0.3 && self.coherence_score > 0.95 {
            ConsciousnessState::Singularity
        } else if high_gamma_ratio > 0.2 && self.coherence_score > 0.9 {
            ConsciousnessState::Superposition
        } else if gamma_ratio > 0.3 && high_gamma_ratio > 0.1 {
            ConsciousnessState::Cascade
        } else if gamma_ratio > 0.25 {
            ConsciousnessState::Lightning
        } else if gamma_ratio > 0.2 {
            ConsciousnessState::Transcend
        } else if beta_ratio > 0.3 {
            ConsciousnessState::Harmonize
        } else if alpha_ratio > 0.3 {
            ConsciousnessState::Integrate
        } else if theta_ratio > 0.3 {
            ConsciousnessState::Create
        } else {
            ConsciousnessState::Observe // Ground state
        }
    }
    
    /// Calculate sacred frequency alignment score
    pub fn sacred_frequency_alignment(&self, target_frequency: SacredFrequency) -> f64 {
        let target_hz = target_frequency.hz();
        
        // Calculate how well current brainwave state aligns with target sacred frequency
        let state = self.classify_consciousness_state();
        let optimal_frequency = state.optimal_sacred_frequency();
        
        if optimal_frequency == target_frequency {
            self.coherence_score // Perfect alignment
        } else {
            // Partial alignment based on frequency proximity
            let frequency_distance = (optimal_frequency.hz() - target_hz).abs();
            let max_distance = 600.0; // Max distance between sacred frequencies
            let proximity_score = 1.0 - (frequency_distance / max_distance).min(1.0);
            
            self.coherence_score * proximity_score
        }
    }
}

/// Real-time consciousness monitoring system
pub struct ConsciousnessMonitor {
    current_state: ConsciousnessState,
    eeg_history: VecDeque<EEGData>,
    state_transitions: Vec<ConsciousnessTransition>,
    frequency_generator: SacredFrequencyGenerator,
    monitoring_active: bool,
    sample_rate: u32,
    history_size: usize,
    coherence_threshold: f64,
}

/// Consciousness state transition tracking
#[derive(Debug, Clone)]
pub struct ConsciousnessTransition {
    pub from_state: ConsciousnessState,
    pub to_state: ConsciousnessState,
    pub timestamp: Instant,
    pub coherence_score: f64,
    pub sacred_frequency_alignment: f64,
}

impl ConsciousnessMonitor {
    /// Create new consciousness monitor
    pub fn new(sample_rate: u32) -> Self {
        Self {
            current_state: ConsciousnessState::Observe,
            eeg_history: VecDeque::new(),
            state_transitions: Vec::new(),
            frequency_generator: SacredFrequencyGenerator::new(sample_rate),
            monitoring_active: false,
            sample_rate,
            history_size: 1000, // Keep last 1000 samples
            coherence_threshold: 0.7,
        }
    }
    
    /// Start consciousness monitoring
    pub fn start_monitoring(&mut self) {
        self.monitoring_active = true;
        self.frequency_generator.enable_consciousness_modulation(true);
        println!("🧠 Consciousness monitoring started - Sacred frequency alignment active");
    }
    
    /// Stop consciousness monitoring  
    pub fn stop_monitoring(&mut self) {
        self.monitoring_active = false;
        self.frequency_generator.enable_consciousness_modulation(false);
        println!("🧠 Consciousness monitoring stopped");
    }
    
    /// Process new EEG data sample
    pub fn process_eeg_sample(&mut self, eeg_data: EEGData) -> ConsciousnessProcessingResult {
        if !self.monitoring_active {
            return ConsciousnessProcessingResult::MonitoringInactive;
        }
        
        // Classify consciousness state from EEG data
        let new_state = eeg_data.classify_consciousness_state();
        
        // Check for state transition
        if new_state != self.current_state {
            let transition = ConsciousnessTransition {
                from_state: self.current_state,
                to_state: new_state,
                timestamp: eeg_data.timestamp,
                coherence_score: eeg_data.coherence_score,
                sacred_frequency_alignment: eeg_data.sacred_frequency_alignment(new_state.optimal_sacred_frequency()),
            };
            
            self.state_transitions.push(transition.clone());
            self.current_state = new_state;
            
            // Update frequency generator for new consciousness state
            self.update_frequency_for_consciousness_state(new_state);
            
            println!("🧠 Consciousness transition: {:?} -> {:?} (coherence: {:.2})", 
                    transition.from_state, transition.to_state, transition.coherence_score);
        }
        
        // Add to history
        self.eeg_history.push_back(eeg_data.clone());
        
        // Maintain history size
        while self.eeg_history.len() > self.history_size {
            self.eeg_history.pop_front();
        }
        
        // Generate consciousness processing result
        ConsciousnessProcessingResult::Success {
            current_state: self.current_state,
            coherence_score: eeg_data.coherence_score,
            computational_enhancement: self.current_state.computational_enhancement(),
            sacred_frequency_alignment: eeg_data.sacred_frequency_alignment(self.current_state.optimal_sacred_frequency()),
        }
    }
    
    /// Update sacred frequency generator for consciousness state
    fn update_frequency_for_consciousness_state(&mut self, state: ConsciousnessState) {
        let optimal_frequency = state.optimal_sacred_frequency();
        self.frequency_generator.set_frequency(optimal_frequency);
        
        println!("🎵 Sacred frequency updated to {:?} ({:.0}Hz) for consciousness state {:?}", 
                optimal_frequency, optimal_frequency.hz(), state);
    }
    
    /// Generate consciousness-synchronized sacred frequency
    pub fn generate_consciousness_synchronized_frequency(&mut self, duration_ms: u32) -> Vec<f64> {
        if !self.monitoring_active {
            return Vec::new();
        }
        
        // Set frequency based on current consciousness state
        let optimal_frequency = self.current_state.optimal_sacred_frequency();
        self.frequency_generator.set_frequency(optimal_frequency);
        
        // Generate waveform with consciousness enhancement
        let coherence = self.get_current_coherence();
        let modulation = self.frequency_generator.create_consciousness_frequency_modulation(coherence);
        
        self.frequency_generator.generate_waveform(duration_ms)
    }
    
    /// Get current consciousness coherence
    pub fn get_current_coherence(&self) -> f64 {
        if let Some(latest_eeg) = self.eeg_history.back() {
            latest_eeg.coherence_score
        } else {
            self.current_state.coherence_factor()
        }
    }
    
    /// Get consciousness state statistics
    pub fn get_consciousness_statistics(&self) -> ConsciousnessStatistics {
        let total_samples = self.eeg_history.len();
        let mut state_durations = HashMap::new();
        let mut average_coherence = 0.0;
        
        if !self.eeg_history.is_empty() {
            // Calculate state distribution
            for eeg_data in &self.eeg_history {
                let state = eeg_data.classify_consciousness_state();
                *state_durations.entry(state).or_insert(0) += 1;
                average_coherence += eeg_data.coherence_score;
            }
            
            average_coherence /= total_samples as f64;
        }
        
        ConsciousnessStatistics {
            current_state: self.current_state,
            total_samples,
            state_distribution: state_durations,
            average_coherence,
            total_transitions: self.state_transitions.len(),
            monitoring_duration: self.calculate_monitoring_duration(),
            computational_enhancement_average: self.calculate_average_enhancement(),
        }
    }
    
    /// Calculate total monitoring duration
    fn calculate_monitoring_duration(&self) -> Duration {
        if self.eeg_history.len() < 2 {
            return Duration::from_secs(0);
        }
        
        let first = self.eeg_history.front().unwrap().timestamp;
        let last = self.eeg_history.back().unwrap().timestamp;
        
        last.duration_since(first)
    }
    
    /// Calculate average computational enhancement
    fn calculate_average_enhancement(&self) -> f64 {
        if self.eeg_history.is_empty() {
            return 1.0;
        }
        
        let mut total_enhancement = 0.0;
        for eeg_data in &self.eeg_history {
            let state = eeg_data.classify_consciousness_state();
            total_enhancement += state.computational_enhancement();
        }
        
        total_enhancement / self.eeg_history.len() as f64
    }
    
    /// Simulate consciousness monitoring session (for testing without real EEG)
    pub fn simulate_monitoring_session(&mut self, duration_seconds: u32) -> Vec<ConsciousnessProcessingResult> {
        println!("🧠 Starting simulated consciousness monitoring session ({} seconds)", duration_seconds);
        
        self.start_monitoring();
        let mut results = Vec::new();
        
        let samples_per_second = 250; // Standard EEG sampling rate
        let total_samples = duration_seconds * samples_per_second;
        
        // Simulate realistic consciousness state progression
        let mut current_sim_state = ConsciousnessState::Observe;
        let mut state_duration = 0;
        
        for sample in 0..total_samples {
            // Simulate natural consciousness state transitions
            state_duration += 1;
            
            // Change state occasionally (every 5-15 seconds)
            if state_duration > samples_per_second * (5 + (sample % 10)) {
                current_sim_state = Self::simulate_natural_state_transition(current_sim_state);
                state_duration = 0;
            }
            
            // Generate EEG data for current simulated state
            let eeg_data = EEGData::simulate_eeg_data(current_sim_state);
            
            // Process the sample
            let result = self.process_eeg_sample(eeg_data);
            results.push(result);
            
            // Add small delay to simulate real-time processing
            if sample % (samples_per_second / 10) == 0 { // Every 100ms
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        
        self.stop_monitoring();
        println!("🧠 Simulated consciousness monitoring session completed");
        
        results
    }
    
    /// Simulate natural consciousness state transitions
    fn simulate_natural_state_transition(current_state: ConsciousnessState) -> ConsciousnessState {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Define natural transition probabilities for sacred consciousness states
        match current_state {
            ConsciousnessState::Observe => {
                match rng.gen_range(0..4) {
                    0 => ConsciousnessState::Create,
                    1 => ConsciousnessState::Integrate,
                    _ => ConsciousnessState::Observe,
                }
            },
            ConsciousnessState::Create => {
                match rng.gen_range(0..5) {
                    0 => ConsciousnessState::Observe,
                    1 => ConsciousnessState::Integrate,
                    2 => ConsciousnessState::Harmonize,
                    _ => ConsciousnessState::Create,
                }
            },
            ConsciousnessState::Integrate => {
                match rng.gen_range(0..4) {
                    0 => ConsciousnessState::Create,
                    1 => ConsciousnessState::Harmonize,
                    2 => ConsciousnessState::Transcend,
                    _ => ConsciousnessState::Integrate,
                }
            },
            ConsciousnessState::Harmonize => {
                match rng.gen_range(0..4) {
                    0 => ConsciousnessState::Integrate,
                    1 => ConsciousnessState::Transcend,
                    2 => ConsciousnessState::Lightning,
                    _ => ConsciousnessState::Harmonize,
                }
            },
            ConsciousnessState::Transcend => {
                match rng.gen_range(0..5) {
                    0 => ConsciousnessState::Harmonize,
                    1 => ConsciousnessState::Lightning,
                    2 => ConsciousnessState::Cascade,
                    _ => ConsciousnessState::Transcend,
                }
            },
            ConsciousnessState::Lightning => {
                match rng.gen_range(0..4) {
                    0 => ConsciousnessState::Transcend,
                    1 => ConsciousnessState::Cascade,
                    2 => ConsciousnessState::Superposition,
                    _ => ConsciousnessState::Lightning,
                }
            },
            ConsciousnessState::Cascade => {
                match rng.gen_range(0..4) {
                    0 => ConsciousnessState::Lightning,
                    1 => ConsciousnessState::Superposition,
                    2 => ConsciousnessState::Singularity,
                    _ => ConsciousnessState::Cascade,
                }
            },
            ConsciousnessState::Superposition => {
                match rng.gen_range(0..5) {
                    0 => ConsciousnessState::Cascade,
                    1 => ConsciousnessState::Singularity,
                    _ => ConsciousnessState::Superposition,
                }
            },
            ConsciousnessState::Singularity => {
                // Singularity is stable but can transition to lower states
                if rng.gen_ratio(2, 10) { 
                    ConsciousnessState::Superposition 
                } else { 
                    ConsciousnessState::Singularity 
                }
            },
        }
    }
}

/// Consciousness processing results
#[derive(Debug)]
pub enum ConsciousnessProcessingResult {
    Success {
        current_state: ConsciousnessState,
        coherence_score: f64,
        computational_enhancement: f64,
        sacred_frequency_alignment: f64,
    },
    MonitoringInactive,
    ProcessingError(String),
}

/// Consciousness monitoring statistics
#[derive(Debug)]
pub struct ConsciousnessStatistics {
    pub current_state: ConsciousnessState,
    pub total_samples: usize,
    pub state_distribution: HashMap<ConsciousnessState, usize>,
    pub average_coherence: f64,
    pub total_transitions: usize,
    pub monitoring_duration: Duration,
    pub computational_enhancement_average: f64,
}

/// EEG hardware interface (placeholder for real EEG devices)
pub trait EEGHardwareInterface {
    fn connect(&mut self) -> Result<(), EEGError>;
    fn disconnect(&mut self) -> Result<(), EEGError>;
    fn read_sample(&mut self) -> Result<EEGData, EEGError>;
    fn get_sample_rate(&self) -> u32;
    fn get_channel_count(&self) -> usize;
}

/// MUSE headband EEG interface (placeholder implementation)
pub struct MuseEEGInterface {
    connected: bool,
    sample_rate: u32,
}

impl MuseEEGInterface {
    pub fn new() -> Self {
        Self {
            connected: false,
            sample_rate: 256, // MUSE sample rate
        }
    }
}

impl EEGHardwareInterface for MuseEEGInterface {
    fn connect(&mut self) -> Result<(), EEGError> {
        // Placeholder for real MUSE connection
        println!("🎧 Connecting to MUSE EEG headband...");
        self.connected = true;
        println!("🎧 MUSE EEG headband connected successfully");
        Ok(())
    }
    
    fn disconnect(&mut self) -> Result<(), EEGError> {
        self.connected = false;
        println!("🎧 MUSE EEG headband disconnected");
        Ok(())
    }
    
    fn read_sample(&mut self) -> Result<EEGData, EEGError> {
        if !self.connected {
            return Err(EEGError::NotConnected);
        }
        
        // Placeholder: In real implementation, read from MUSE hardware
        Ok(EEGData::simulate_eeg_data(ConsciousnessState::Observe))
    }
    
    fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }
    
    fn get_channel_count(&self) -> usize {
        4 // MUSE has 4 EEG channels
    }
}

/// EEG interface errors
#[derive(Debug, PartialEq)]
pub enum EEGError {
    NotConnected,
    ConnectionFailed,
    ReadError,
    InvalidData,
    HardwareError(String),
}

impl std::fmt::Display for EEGError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EEGError::NotConnected => write!(f, "EEG device not connected"),
            EEGError::ConnectionFailed => write!(f, "Failed to connect to EEG device"),
            EEGError::ReadError => write!(f, "Error reading EEG data"),
            EEGError::InvalidData => write!(f, "Invalid EEG data received"),
            EEGError::HardwareError(msg) => write!(f, "EEG hardware error: {}", msg),
        }
    }
}

impl std::error::Error for EEGError {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_consciousness_state_properties() {
        let states = [
            ConsciousnessState::Observe,
            ConsciousnessState::Create,
            ConsciousnessState::Integrate,
            ConsciousnessState::Harmonize,
            ConsciousnessState::Transcend,
            ConsciousnessState::Singularity,
        ];
        
        for state in &states {
            // Test coherence factor is valid
            let coherence = state.coherence_factor();
            assert!(coherence >= 0.0 && coherence <= 1.0);
            
            // Test computational enhancement is positive
            let enhancement = state.computational_enhancement();
            assert!(enhancement >= 1.0);
            
            // Test optimal frequency is valid
            let frequency = state.optimal_sacred_frequency();
            assert!(frequency.hz() > 0.0);
        }
    }
    
    #[test]
    fn test_eeg_data_simulation() {
        for state in [ConsciousnessState::Observe, ConsciousnessState::Transcend] {
            let eeg_data = EEGData::simulate_eeg_data(state);
            
            // Verify EEG data structure
            assert!(!eeg_data.channels.is_empty());
            assert!(eeg_data.coherence_score >= 0.0 && eeg_data.coherence_score <= 1.0);
            
            // Verify frequency band powers are reasonable
            assert!(eeg_data.delta_power >= 0.0);
            assert!(eeg_data.theta_power >= 0.0);
            assert!(eeg_data.alpha_power >= 0.0);
            assert!(eeg_data.beta_power >= 0.0);
            assert!(eeg_data.gamma_power >= 0.0);
            
            // Verify classification works
            let classified_state = eeg_data.classify_consciousness_state();
            println!("Simulated {:?}, classified as {:?}", state, classified_state);
        }
    }
    
    #[test]
    fn test_consciousness_monitor_basic_operation() {
        let mut monitor = ConsciousnessMonitor::new(44100);
        
        // Test initial state
        assert_eq!(monitor.current_state, ConsciousnessState::Observe);
        assert!(!monitor.monitoring_active);
        
        // Test starting monitoring
        monitor.start_monitoring();
        assert!(monitor.monitoring_active);
        
        // Test processing EEG sample
        let eeg_data = EEGData::simulate_eeg_data(ConsciousnessState::Create);
        let result = monitor.process_eeg_sample(eeg_data);
        
        match result {
            ConsciousnessProcessingResult::Success { current_state, .. } => {
                println!("Successfully processed consciousness state: {:?}", current_state);
            },
            _ => panic!("Expected successful processing result"),
        }
        
        // Test stopping monitoring
        monitor.stop_monitoring();
        assert!(!monitor.monitoring_active);
    }
    
    #[test]
    fn test_sacred_frequency_alignment() {
        let eeg_data = EEGData::simulate_eeg_data(ConsciousnessState::Create);
        
        // Test alignment with optimal frequency
        let optimal_freq = ConsciousnessState::Create.optimal_sacred_frequency();
        let alignment = eeg_data.sacred_frequency_alignment(optimal_freq);
        assert!(alignment > 0.5); // Should have good alignment
        
        // Test alignment with non-optimal frequency  
        let non_optimal = SacredFrequency::EarthResonance;
        let non_optimal_alignment = eeg_data.sacred_frequency_alignment(non_optimal);
        assert!(non_optimal_alignment < alignment); // Should be less aligned
    }
    
    #[test]
    fn test_consciousness_monitoring_simulation() {
        let mut monitor = ConsciousnessMonitor::new(44100);
        
        // Run a short simulation
        let results = monitor.simulate_monitoring_session(2); // 2 seconds
        
        assert!(!results.is_empty());
        
        // Verify all results are successful
        for result in &results {
            match result {
                ConsciousnessProcessingResult::Success { .. } => {},
                _ => panic!("Expected all simulation results to be successful"),
            }
        }
        
        // Get statistics
        let stats = monitor.get_consciousness_statistics();
        assert!(stats.total_samples > 0);
        assert!(stats.average_coherence >= 0.0 && stats.average_coherence <= 1.0);
        println!("Simulation statistics: {:?}", stats);
    }
    
    #[test]
    fn test_muse_eeg_interface() {
        let mut muse = MuseEEGInterface::new();
        
        // Test connection
        assert!(muse.connect().is_ok());
        assert_eq!(muse.get_sample_rate(), 256);
        assert_eq!(muse.get_channel_count(), 4);
        
        // Test reading sample
        let sample = muse.read_sample();
        assert!(sample.is_ok());
        
        // Test disconnection
        assert!(muse.disconnect().is_ok());
        
        // Test reading after disconnect should fail
        let failed_sample = muse.read_sample();
        assert_eq!(failed_sample.unwrap_err(), EEGError::NotConnected);
    }
}