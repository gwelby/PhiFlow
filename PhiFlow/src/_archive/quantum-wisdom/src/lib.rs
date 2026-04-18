use num_complex::Complex64;
use ndarray::Array2;
use quantum_utils::{PHI, sacred_ratio};
use quantum_sacred::SacredGeometry;

pub const WISDOM_GROUND: f64 = 432.0;
pub const WISDOM_CREATE: f64 = 528.0;
pub const WISDOM_HEART: f64 = 594.0;
pub const WISDOM_VOICE: f64 = 672.0;
pub const WISDOM_UNITY: f64 = 768.0;

#[derive(Debug, Clone)]
pub struct QuantumWisdom {
    wisdom_field: Array2<Complex64>,
    sacred_geometry: SacredGeometry,
    frequency: f64,
    coherence: f64,
}

impl QuantumWisdom {
    pub fn new(frequency: f64) -> Self {
        let size = (PHI * 21.0).round() as usize;
        let wisdom_field = Array2::zeros((size, size));
        
        Self {
            wisdom_field,
            sacred_geometry: SacredGeometry::new(frequency),
            frequency,
            coherence: 1.0,
        }
    }

    pub fn ground() -> Self {
        Self::new(WISDOM_GROUND)
    }

    pub fn create() -> Self {
        Self::new(WISDOM_CREATE)
    }

    pub fn heart() -> Self {
        Self::new(WISDOM_HEART)
    }

    pub fn voice() -> Self {
        Self::new(WISDOM_VOICE)
    }

    pub fn unity() -> Self {
        Self::new(WISDOM_UNITY)
    }

    pub fn evolve(&mut self, time_step: f64) {
        let phi_factor = sacred_ratio((time_step * self.frequency / WISDOM_GROUND) as u32);
        
        let size = self.wisdom_field.dim().0;
        for i in 0..size {
            for j in 0..size {
                let wisdom_factor = phi_factor * (i + j) as f64 / size as f64;
                self.wisdom_field[[i, j]] *= Complex64::from_polar(1.0, wisdom_factor);
            }
        }

        self.sacred_geometry.harmonize_all();
    }

    pub fn get_wisdom_field(&self) -> &Array2<Complex64> {
        &self.wisdom_field
    }

    pub fn get_frequency(&self) -> f64 {
        self.frequency
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }

    pub fn set_coherence(&mut self, coherence: f64) {
        self.coherence = coherence.max(0.0).min(1.0);
    }
}
