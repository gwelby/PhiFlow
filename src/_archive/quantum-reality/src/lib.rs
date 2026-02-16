use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use quantum_dimension::{QuantumDimension, PHI, GROUND_FREQUENCY, CREATE_FREQUENCY, UNITY_FREQUENCY};
use quantum_sacred::SacredGeometry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumReality {
    dimensions: Vec<QuantumDimension>,
    sacred_geometry: SacredGeometry,
    reality_field: Array2<Complex64>,
    coherence: f64,
}

impl QuantumReality {
    pub fn new(base_frequency: f64) -> Self {
        let size = (PHI * 21.0).round() as usize;
        let reality_field = Array2::zeros((size, size));
        
        let dimensions = vec![
            QuantumDimension::ground(),
            QuantumDimension::create(),
            QuantumDimension::unity(),
        ];

        let sacred_geometry = SacredGeometry::new(base_frequency);

        Self {
            dimensions,
            sacred_geometry,
            reality_field,
            coherence: 1.0,
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

    pub fn evolve(&mut self, time_step: f64) {
        for dim in &mut self.dimensions {
            dim.evolve(time_step);
        }

        self.sacred_geometry.harmonize_all();
        
        let sacred_field = self.sacred_geometry.get_coherence_field();
        let size = self.reality_field.dim().0;
        
        for i in 0..size {
            for j in 0..size {
                let phi_factor = PHI.powi((i + j) as i32);
                let sacred_value = sacred_field[[i % sacred_field.dim().0, j % sacred_field.dim().1]];
                self.reality_field[[i, j]] = sacred_value * phi_factor * self.coherence;
            }
        }
    }

    pub fn set_coherence(&mut self, coherence: f64) {
        self.coherence = coherence.max(0.0).min(1.0);
        for dim in &mut self.dimensions {
            dim.set_coherence(self.coherence);
        }
    }

    pub fn get_dimensions(&self) -> &[QuantumDimension] {
        &self.dimensions
    }

    pub fn get_sacred_geometry(&self) -> &SacredGeometry {
        &self.sacred_geometry
    }

    pub fn get_reality_field(&self) -> &Array2<Complex64> {
        &self.reality_field
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }
}
