use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use crate::quantum::quantum_core::{QuantumCore, QuantumState};
use crate::quantum::quantum_harmonizer::QuantumHarmonizer;

/// QuantumMonitor - Real-time quantum field monitoring and coherence maintenance
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumMonitor {
    // Core quantum systems
    core: Arc<RwLock<QuantumCore>>,
    harmonizer: Arc<RwLock<QuantumHarmonizer>>,
    
    // Monitoring metrics
    coherence_history: Vec<CoherenceMetric>,
    frequency_stability: f64,
    phi_alignment: f64,
    dimensional_balance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceMetric {
    timestamp: DateTime<Utc>,
    frequency: f64,
    coherence: f64,
    phi_level: f64,
    pattern: String,
}

impl QuantumMonitor {
    pub fn new(core: Arc<RwLock<QuantumCore>>) -> Self {
        Self {
            core,
            harmonizer: Arc::new(RwLock::new(QuantumHarmonizer::new())),
            coherence_history: Vec::new(),
            frequency_stability: 1.0,
            phi_alignment: 1.618034,
            dimensional_balance: 1.0,
        }
    }

    /// Monitor quantum coherence in real-time
    pub async fn monitor_coherence(&mut self) -> CoherenceMetric {
        let core = self.core.read();
        let state = core.current_state.clone();
        
        // Calculate current metrics
        let metric = CoherenceMetric {
            timestamp: Utc::now(),
            frequency: state.frequency,
            coherence: state.coherence,
            phi_level: self.calculate_phi_level(&state),
            pattern: format!("{:?}", state.pattern),
        };
        
        // Update history
        self.coherence_history.push(metric.clone());
        
        // Maintain optimal coherence
        self.maintain_coherence().await;
        
        metric
    }

    /// Calculate phi alignment level
    fn calculate_phi_level(&self, state: &QuantumState) -> f64 {
        let phi = 1.618034;
        let freq_ratio = state.frequency / 432.0;
        let phi_ratio = freq_ratio / phi;
        
        // Return alignment level (1.0 = perfect)
        (phi_ratio - phi_ratio.floor()).abs()
    }

    /// Maintain optimal coherence
    async fn maintain_coherence(&mut self) {
        let mut core = self.core.write();
        let mut harmonizer = self.harmonizer.write();
        
        // Check frequency stability
        let current_freq = core.current_state.frequency;
        let target_freq = self.find_nearest_harmonic(current_freq);
        
        if (current_freq - target_freq).abs() > 1.0 {
            // Adjust frequency
            core.set_frequency(target_freq);
            
            // Generate new sacred geometry
            let merkaba = harmonizer.generate_merkaba(target_freq);
            println!("⚡ Regenerated Merkaba field at {} Hz", target_freq);
        }
        
        // Update stability metrics
        self.update_stability_metrics();
    }

    /// Find nearest harmonic frequency
    fn find_nearest_harmonic(&self, freq: f64) -> f64 {
        let harmonics = vec![432.0, 528.0, 594.0, 672.0, 720.0, 768.0];
        
        harmonics.iter()
            .min_by(|&&a, &b| {
                (a - freq).abs().partial_cmp(&(b - freq).abs()).unwrap()
            })
            .copied()
            .unwrap_or(432.0)
    }

    /// Update stability metrics
    fn update_stability_metrics(&mut self) {
        if let Some(recent_metrics) = self.coherence_history.last() {
            // Calculate frequency stability
            self.frequency_stability = recent_metrics.coherence;
            
            // Update phi alignment
            self.phi_alignment = recent_metrics.phi_level;
            
            // Calculate dimensional balance
            self.dimensional_balance = self.calculate_dimensional_balance();
        }
    }

    /// Calculate dimensional balance
    fn calculate_dimensional_balance(&self) -> f64 {
        let phi = 1.618034;
        let recent_metrics: Vec<_> = self.coherence_history.iter()
            .rev()
            .take(5)
            .collect();
            
        let balance = recent_metrics.iter()
            .map(|m| m.coherence)
            .sum::<f64>() / recent_metrics.len() as f64;
            
        (balance * phi).min(1.0)
    }

    /// Get current stability metrics
    pub fn get_stability_metrics(&self) -> (f64, f64, f64) {
        (
            self.frequency_stability,
            self.phi_alignment,
            self.dimensional_balance
        )
    }

    /// Get coherence history
    pub fn get_coherence_history(&self) -> &[CoherenceMetric] {
        &self.coherence_history
    }
}
