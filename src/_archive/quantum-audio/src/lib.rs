use anyhow::Result;
use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};

pub mod quantum_constants;

pub struct QuantumAudio {
    sample_rate: f64,
    buffer_size: usize,
    frequencies: Vec<f64>,
    amplitudes: Vec<f64>,
    phases: Vec<f64>,
}

impl QuantumAudio {
    pub fn new(sample_rate: f64, buffer_size: usize) -> Self {
        Self {
            sample_rate,
            buffer_size,
            frequencies: vec![
                quantum_constants::GROUND_FREQUENCY,
                quantum_constants::CREATE_FREQUENCY,
                quantum_constants::HEART_FREQUENCY,
                quantum_constants::VOICE_FREQUENCY,
                quantum_constants::VISION_FREQUENCY,
                quantum_constants::UNITY_FREQUENCY,
            ],
            amplitudes: vec![
                1.0,
                quantum_constants::PHI,
                quantum_constants::PHI_SQUARED,
                quantum_constants::PHI_CUBED,
                quantum_constants::PHI_SQUARED * quantum_constants::PHI,
                quantum_constants::PHI_CUBED * quantum_constants::PHI,
            ],
            phases: vec![0.0; 6],
        }
    }

    pub fn generate_sample(&mut self, t: f64) -> f64 {
        self.frequencies.iter()
            .zip(self.amplitudes.iter())
            .zip(self.phases.iter())
            .map(|((&f, &a), &p)| a * (2.0 * std::f64::consts::PI * f * t + p).sin())
            .sum()
    }

    pub fn generate_buffer(&mut self) -> Vec<f64> {
        let mut buffer = Vec::with_capacity(self.buffer_size);
        let dt = 1.0 / self.sample_rate;
        
        for i in 0..self.buffer_size {
            let t = i as f64 * dt;
            buffer.push(self.generate_sample(t));
        }
        
        buffer
    }

    pub fn set_frequency(&mut self, index: usize, frequency: f64) {
        if index < self.frequencies.len() {
            self.frequencies[index] = frequency;
        }
    }

    pub fn set_amplitude(&mut self, index: usize, amplitude: f64) {
        if index < self.amplitudes.len() {
            self.amplitudes[index] = amplitude;
        }
    }

    pub fn set_phase(&mut self, index: usize, phase: f64) {
        if index < self.phases.len() {
            self.phases[index] = phase;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_audio() {
        let mut audio = QuantumAudio::new(432000.0, 1024);
        let buffer = audio.generate_buffer();
        
        assert_eq!(buffer.len(), 1024);
        
        // Check that the buffer contains non-zero values
        assert!(buffer.iter().any(|&x| x != 0.0));
    }
}
