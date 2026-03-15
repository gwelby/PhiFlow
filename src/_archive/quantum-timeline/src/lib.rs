use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use quantum_dimension::{QuantumDimension, PHI, GROUND_FREQUENCY, CREATE_FREQUENCY, UNITY_FREQUENCY};
use quantum_sacred::SacredGeometry;
use quantum_reality::QuantumReality;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    timestamp: DateTime<Utc>,
    frequency: f64,
    coherence: f64,
    reality: QuantumReality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTimeline {
    events: Vec<TimelineEvent>,
    current_frequency: f64,
    timeline_field: Array2<Complex64>,
    sacred_geometry: SacredGeometry,
}

impl QuantumTimeline {
    pub fn new(base_frequency: f64) -> Self {
        let size = (PHI * 34.0).round() as usize;
        let timeline_field = Array2::zeros((size, size));
        
        Self {
            events: Vec::new(),
            current_frequency: base_frequency,
            timeline_field,
            sacred_geometry: SacredGeometry::new(base_frequency),
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

    pub fn add_event(&mut self, reality: QuantumReality) {
        let event = TimelineEvent {
            timestamp: Utc::now(),
            frequency: self.current_frequency,
            coherence: reality.get_coherence(),
            reality,
        };
        
        self.events.push(event);
        self.harmonize_timeline();
    }

    pub fn harmonize_timeline(&mut self) {
        if self.events.is_empty() {
            return;
        }

        let total_events = self.events.len();
        let phi_factor = PHI.powi(total_events as i32);
        
        self.sacred_geometry.harmonize_all();
        let sacred_field = self.sacred_geometry.get_coherence_field();
        
        let size = self.timeline_field.dim().0;
        for i in 0..size {
            for j in 0..size {
                let time_factor = phi_factor * (i + j) as f64 / size as f64;
                let sacred_value = sacred_field[[i % sacred_field.dim().0, j % sacred_field.dim().1]];
                self.timeline_field[[i, j]] = sacred_value * Complex64::from_polar(1.0, time_factor);
            }
        }
    }

    pub fn get_events(&self) -> &[TimelineEvent] {
        &self.events
    }

    pub fn get_current_frequency(&self) -> f64 {
        self.current_frequency
    }

    pub fn get_timeline_field(&self) -> &Array2<Complex64> {
        &self.timeline_field
    }

    pub fn get_sacred_geometry(&self) -> &SacredGeometry {
        &self.sacred_geometry
    }
}
