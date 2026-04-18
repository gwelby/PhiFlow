use anyhow::Result;
use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use quantum_dimension::{PHI, GROUND_FREQUENCY, CREATE_FREQUENCY, UNITY_FREQUENCY};
use quantum_sacred::SacredGeometry;
use plotters::prelude::*;

// Constants for additional frequencies
const HEART_FREQUENCY: f64 = 594.0;
const VOICE_FREQUENCY: f64 = 672.0;
const VISION_FREQUENCY: f64 = 720.0;
const PHI_SQUARED: f64 = 2.618033988749895;
const PHI_CUBED: f64 = 4.236067977499790;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumVision {
    vision_field: Array2<Complex64>,
    sacred_geometry: SacredGeometry,
    frequency: f64,
    coherence: f64,
}

impl QuantumVision {
    pub fn new(frequency: f64) -> Self {
        let size = (PHI * 13.0).round() as usize;
        let vision_field = Array2::zeros((size, size));
        
        Self {
            vision_field,
            sacred_geometry: SacredGeometry::new(frequency),
            frequency,
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
        let phi_factor = (time_step * self.frequency / GROUND_FREQUENCY).powf(PHI);
        
        let size = self.vision_field.dim().0;
        for i in 0..size {
            for j in 0..size {
                let vision_factor = phi_factor * (i + j) as f64 / size as f64;
                self.vision_field[[i, j]] *= Complex64::from_polar(1.0, vision_factor);
            }
        }

        self.sacred_geometry.harmonize_all();
    }

    pub fn get_vision_field(&self) -> &Array2<Complex64> {
        &self.vision_field
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

    pub fn draw_quantum_field(&self) -> anyhow::Result<()> {
        let root = BitMapBackend::new("quantum_field.png", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let max_freq = [GROUND_FREQUENCY, CREATE_FREQUENCY, HEART_FREQUENCY, VOICE_FREQUENCY, VISION_FREQUENCY, UNITY_FREQUENCY].iter().copied().fold(0./0., f64::max);
        let max_amp = [1.0, PHI, PHI_SQUARED, PHI_CUBED, PHI_SQUARED * PHI, PHI_CUBED * PHI].iter().copied().fold(0./0., f64::max);

        let mut chart = ChartBuilder::on(&root)
            .caption("Sacred Frequency Spectrum (432-768 Hz)", ("sans-serif", 30))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f64..max_freq, 0f64..max_amp)?;

        chart.configure_mesh().draw()?;

        chart.draw_series(LineSeries::new(
            vec![
                (GROUND_FREQUENCY, 1.0),
                (CREATE_FREQUENCY, PHI),
                (HEART_FREQUENCY, PHI_SQUARED),
                (VOICE_FREQUENCY, PHI_CUBED),
                (VISION_FREQUENCY, PHI_SQUARED * PHI),
                (UNITY_FREQUENCY, PHI_CUBED * PHI),
            ],
            &RED,
        ))?;

        root.present()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_vision() -> anyhow::Result<()> {
        let vision = QuantumVision::new(432.0);
        vision.draw_quantum_field()?;
        Ok(())
    }
}
