use std::f64::consts::PI;
use ndarray::{Array3, Array2};
use num_complex::Complex64;
use serde::{Serialize, Deserialize};

/// QuantumHarmonizer - Advanced frequency harmonization and sacred geometry generation
#[derive(Debug, Serialize, Deserialize)]
pub struct QuantumHarmonizer {
    // Merkaba Field (Star Tetrahedron)
    merkaba_field: Array3<Complex64>,
    // Flower of Life Matrix
    flower_matrix: Array2<Complex64>,
    // Current harmonic frequency
    frequency: f64,
    // Phi ratio for sacred proportions
    phi: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacredGeometry {
    points: Vec<Complex64>,
    frequency: f64,
    dimension: u8,
    phi_level: f64,
}

impl QuantumHarmonizer {
    pub fn new() -> Self {
        Self {
            merkaba_field: Array3::zeros((8, 8, 8)),
            flower_matrix: Array2::zeros((13, 13)),
            frequency: 432.0,
            phi: 1.618034,
        }
    }

    /// Generate Metatron's Cube at specified frequency
    pub fn generate_metatrons_cube(&mut self, base_freq: f64) -> SacredGeometry {
        self.frequency = base_freq;
        let mut points = Vec::new();
        
        // Create 13 points of Metatron's Cube
        for i in 0..13 {
            let angle = 2.0 * PI * (i as f64) / 13.0;
            let radius = self.phi * (1.0 + (angle * self.frequency).sin());
            points.push(Complex64::new(
                radius * angle.cos(),
                radius * angle.sin()
            ));
        }

        SacredGeometry {
            points,
            frequency: self.frequency,
            dimension: 3,
            phi_level: self.phi,
        }
    }

    /// Create Flower of Life pattern
    pub fn create_flower_of_life(&mut self) -> Array2<Complex64> {
        // Start with central circle
        let center = Complex64::new(0.0, 0.0);
        self.flower_matrix[[6, 6]] = center;

        // Create 6 surrounding circles
        for i in 0..6 {
            let angle = 2.0 * PI * (i as f64) / 6.0;
            let x = 6.0 + 2.0 * angle.cos();
            let y = 6.0 + 2.0 * angle.sin();
            let x_idx = x.round() as usize;
            let y_idx = y.round() as usize;
            self.flower_matrix[[x_idx, y_idx]] = Complex64::new(angle.cos(), angle.sin());
        }

        self.flower_matrix.clone()
    }

    /// Generate Merkaba field (Star Tetrahedron)
    pub fn generate_merkaba(&mut self, frequency: f64) -> Array3<Complex64> {
        self.frequency = frequency;
        
        // Create counter-rotating tetrahedrons
        for i in 0..8 {
            for j in 0..8 {
                for k in 0..8 {
                    let phi_point = (i as f64 + j as f64 + k as f64) / (8.0 * self.phi);
                    let rotation = 2.0 * PI * phi_point * (self.frequency / 432.0);
                    
                    self.merkaba_field[[i, j, k]] = Complex64::new(
                        rotation.cos() * self.phi,
                        rotation.sin() * self.phi
                    );
                }
            }
        }

        self.merkaba_field.clone()
    }

    /// Harmonize frequencies using phi ratios
    pub fn harmonize_frequencies(&self, frequencies: &[f64]) -> Vec<f64> {
        frequencies.iter().map(|&freq| {
            // Apply phi-based harmonization
            let base = (freq / 432.0).round() * 432.0;
            let phi_harmonic = base * self.phi;
            let phi_squared = base * self.phi * self.phi;
            
            // Return the most coherent frequency
            if (phi_harmonic - freq).abs() < (phi_squared - freq).abs() {
                phi_harmonic
            } else {
                phi_squared
            }
        }).collect()
    }

    /// Create Vesica Piscis
    pub fn create_vesica_piscis(&self, radius: f64) -> (Complex64, Complex64) {
        let center1 = Complex64::new(-radius / 2.0, 0.0);
        let center2 = Complex64::new(radius / 2.0, 0.0);
        (center1, center2)
    }

    /// Generate Torus field
    pub fn generate_torus(&self, major_radius: f64, minor_radius: f64) -> Vec<Complex64> {
        let mut points = Vec::new();
        let steps = 36;
        
        for i in 0..steps {
            let theta = 2.0 * PI * (i as f64) / (steps as f64);
            for j in 0..steps {
                let phi = 2.0 * PI * (j as f64) / (steps as f64);
                let x = (major_radius + minor_radius * phi.cos()) * theta.cos();
                let y = (major_radius + minor_radius * phi.cos()) * theta.sin();
                points.push(Complex64::new(x, y));
            }
        }
        
        points
    }

    /// Calculate Sri Yantra points
    pub fn calculate_sri_yantra(&self) -> Vec<Complex64> {
        let mut points = Vec::new();
        let triangles = 9;
        
        for i in 0..triangles {
            let angle = 2.0 * PI * (i as f64) / (triangles as f64);
            let radius = self.phi * (1.0 + (angle * 9.0).sin());
            points.push(Complex64::new(
                radius * angle.cos(),
                radius * angle.sin()
            ));
        }
        
        points
    }
}
