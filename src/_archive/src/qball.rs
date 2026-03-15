use serde::{Deserialize, Serialize};
use num_complex::Complex64;
use std::f64::consts::PI;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QBall {
    frequency: f64,
    #[serde(skip)]
    state: Complex64,
}

impl QBall {
    pub fn new() -> Self {
        QBall {
            frequency: 528.0, // DNA repair frequency
            state: Complex64::new(1.0, 0.0),
        }
    }

    pub fn entangle(&mut self, other: &mut QBall) {
        let phase = (self.frequency + other.frequency) / 1440.0; // Unity frequency
        self.state *= Complex64::new(phase.cos(), phase.sin());
        other.state *= Complex64::new(phase.cos(), -phase.sin());
        
        // Normalize states
        let norm = (self.state.norm_sqr()).sqrt();
        if norm > 0.0 {
            self.state /= Complex64::new(norm, 0.0);
        }
        
        let norm = (other.state.norm_sqr()).sqrt();
        if norm > 0.0 {
            other.state /= Complex64::new(norm, 0.0);
        }
    }

    pub fn apply_sacred_rotation(&mut self, frequency: f64) {
        let phase = 2.0 * PI * frequency / self.frequency;
        self.state *= Complex64::new(phase.cos(), phase.sin());
        
        // Normalize state
        let norm = (self.state.norm_sqr()).sqrt();
        if norm > 0.0 {
            self.state /= Complex64::new(norm, 0.0);
        }
    }

    pub fn get_frequency(&self) -> f64 {
        self.frequency
    }

    pub fn get_state_magnitude(&self) -> f64 {
        (self.state.norm_sqr()).sqrt()
    }

    pub fn measure_coherence(&self) -> f64 {
        // Coherence is the squared magnitude of the state
        self.state.norm_sqr()
    }
}
