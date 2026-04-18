use num_complex::Complex64;
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::f64::consts::PI;

pub const PHI: f64 = 1.618033988749895;
pub const GROUND_FREQUENCY: f64 = 432.0;
pub const CREATE_FREQUENCY: f64 = 528.0;
pub const UNITY_FREQUENCY: f64 = 768.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDimension {
    frequency: f64,
    coherence: f64,
    phase: Complex64,
    field: Array2<Complex64>,
}

impl QuantumDimension {
    pub fn new(frequency: f64) -> Self {
        let size = (PHI * 8.0).round() as usize;
        let field = Array2::zeros((size, size));
        
        Self {
            frequency,
            coherence: 1.0,
            phase: Complex64::new(0.0, 0.0),
            field,
        }
    }

    pub fn ground() -> Self {
        Self::new(GROUND_FREQUENCY)
    }

    pub fn create() -> Self {
        Self::new(CREATE_FREQUENCY)
    }

    pub fn unity() -> Self {
        Self::new(UNITY_FREQUENCY)
    }

    pub fn set_coherence(&mut self, coherence: f64) {
        self.coherence = coherence.max(0.0).min(1.0);
    }

    pub fn evolve(&mut self, time_step: f64) {
        let omega = 2.0 * PI * self.frequency;
        let phase_shift = Complex64::from_polar(1.0, omega * time_step);
        self.phase *= phase_shift;
        
        let coherence_factor = (self.coherence * PHI).sqrt();
        for elem in self.field.iter_mut() {
            *elem *= phase_shift * coherence_factor;
        }
    }

    pub fn get_frequency(&self) -> f64 {
        self.frequency
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }

    pub fn get_phase(&self) -> Complex64 {
        self.phase
    }

    pub fn get_field(&self) -> &Array2<Complex64> {
        &self.field
    }

    pub fn get_field_mut(&mut self) -> &mut Array2<Complex64> {
        &mut self.field
    }

    pub fn harmonize(&mut self, other: &QuantumDimension) {
        let freq_ratio = self.frequency / other.frequency;
        let harmonic = (freq_ratio * PHI).round() / PHI;
        self.frequency = other.frequency * harmonic;
        self.coherence = (self.coherence * other.coherence).sqrt();
    }
}
