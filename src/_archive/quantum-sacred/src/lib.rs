use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use quantum_dimension::{QuantumDimension, PHI, GROUND_FREQUENCY, CREATE_FREQUENCY, UNITY_FREQUENCY};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacredGeometry {
    dimensions: Vec<QuantumDimension>,
    coherence_field: Array2<Complex64>,
    sacred_frequency: f64,
}

impl SacredGeometry {
    pub fn new(frequency: f64) -> Self {
        let size = (PHI * 13.0).round() as usize;
        let coherence_field = Array2::zeros((size, size));
        
        let dimensions = vec![
            QuantumDimension::ground(),
            QuantumDimension::create(),
            QuantumDimension::unity(),
        ];

        Self {
            dimensions,
            coherence_field,
            sacred_frequency: frequency,
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

    pub fn harmonize_all(&mut self) {
        let mut total_coherence = 1.0;
        for dim in &mut self.dimensions {
            total_coherence *= dim.get_coherence();
            dim.set_coherence(total_coherence.sqrt());
        }

        let size = self.coherence_field.dim().0;
        for i in 0..size {
            for j in 0..size {
                let phi_factor = PHI.powi((i + j) as i32);
                self.coherence_field[[i, j]] = Complex64::from_polar(
                    phi_factor * total_coherence,
                    2.0 * std::f64::consts::PI * self.sacred_frequency
                );
            }
        }
    }

    pub fn get_sacred_frequency(&self) -> f64 {
        self.sacred_frequency
    }

    pub fn get_dimensions(&self) -> &[QuantumDimension] {
        &self.dimensions
    }

    pub fn get_coherence_field(&self) -> &Array2<Complex64> {
        &self.coherence_field
    }
}
