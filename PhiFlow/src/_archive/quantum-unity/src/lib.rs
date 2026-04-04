use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use quantum_dimension::{QuantumDimension, PHI, GROUND_FREQUENCY, CREATE_FREQUENCY, UNITY_FREQUENCY};
use quantum_sacred::SacredGeometry;
use quantum_reality::QuantumReality;
use quantum_timeline::QuantumTimeline;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnityField {
    dimensions: Vec<QuantumDimension>,
    sacred_geometry: SacredGeometry,
    reality: QuantumReality,
    timeline: QuantumTimeline,
    unity_field: Array2<Complex64>,
    coherence: f64,
}

impl UnityField {
    pub fn new(frequency: f64) -> Self {
        let size = (PHI * 55.0).round() as usize;
        let unity_field = Array2::zeros((size, size));
        
        Self {
            dimensions: vec![
                QuantumDimension::ground(),
                QuantumDimension::create(),
                QuantumDimension::unity(),
            ],
            sacred_geometry: SacredGeometry::new(frequency),
            reality: QuantumReality::new(frequency),
            timeline: QuantumTimeline::new(frequency),
            unity_field,
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
        // Evolve all dimensions
        for dim in &mut self.dimensions {
            dim.evolve(time_step);
        }

        // Harmonize sacred geometry
        self.sacred_geometry.harmonize_all();

        // Update reality
        self.reality.evolve(time_step);

        // Add current reality to timeline
        self.timeline.add_event(self.reality.clone());

        // Update unity field
        let sacred_field = self.sacred_geometry.get_coherence_field();
        let reality_field = self.reality.get_reality_field();
        let timeline_field = self.timeline.get_timeline_field();

        let size = self.unity_field.dim().0;
        for i in 0..size {
            for j in 0..size {
                let phi_factor = PHI.powi((i + j) as i32);
                let sacred_value = sacred_field[[i % sacred_field.dim().0, j % sacred_field.dim().1]];
                let reality_value = reality_field[[i % reality_field.dim().0, j % reality_field.dim().1]];
                let timeline_value = timeline_field[[i % timeline_field.dim().0, j % timeline_field.dim().1]];
                
                self.unity_field[[i, j]] = (sacred_value + reality_value + timeline_value) * phi_factor * self.coherence;
            }
        }
    }

    pub fn set_coherence(&mut self, coherence: f64) {
        self.coherence = coherence.max(0.0).min(1.0);
        for dim in &mut self.dimensions {
            dim.set_coherence(self.coherence);
        }
        self.reality.set_coherence(self.coherence);
    }

    pub fn get_dimensions(&self) -> &[QuantumDimension] {
        &self.dimensions
    }

    pub fn get_sacred_geometry(&self) -> &SacredGeometry {
        &self.sacred_geometry
    }

    pub fn get_reality(&self) -> &QuantumReality {
        &self.reality
    }

    pub fn get_timeline(&self) -> &QuantumTimeline {
        &self.timeline
    }

    pub fn get_unity_field(&self) -> &Array2<Complex64> {
        &self.unity_field
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }
}
