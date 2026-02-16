use num_complex::Complex64;
use ndarray::Array2;
use std::f64::consts::PI;

pub const PHI: f64 = 1.618033988749895;
pub const PHI_SQUARED: f64 = 2.618033988749895;
pub const PHI_CUBED: f64 = 4.236067977499790;

pub fn create_phi_matrix(size: usize) -> Array2<Complex64> {
    let mut matrix = Array2::zeros((size, size));
    for i in 0..size {
        for j in 0..size {
            let phi_factor = PHI.powi((i + j) as i32);
            matrix[[i, j]] = Complex64::from_polar(phi_factor, 2.0 * PI * phi_factor);
        }
    }
    matrix
}

pub fn harmonize_frequencies(f1: f64, f2: f64) -> f64 {
    let ratio = f1 / f2;
    let harmonic = (ratio * PHI).round() / PHI;
    f2 * harmonic
}

pub fn calculate_coherence(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    
    let product: f64 = values.iter().product();
    let nth_root = 1.0 / values.len() as f64;
    product.powf(nth_root)
}

pub fn phi_compress(value: f64, level: u32) -> f64 {
    match level {
        0 => value,
        1 => value / PHI,
        2 => value / PHI_SQUARED,
        3 => value / PHI_CUBED,
        _ => value / PHI.powi(level as i32),
    }
}

pub fn phi_expand(value: f64, level: u32) -> f64 {
    match level {
        0 => value,
        1 => value * PHI,
        2 => value * PHI_SQUARED,
        3 => value * PHI_CUBED,
        _ => value * PHI.powi(level as i32),
    }
}

pub fn quantum_resonance(frequency: f64, base_frequency: f64) -> f64 {
    let ratio = frequency / base_frequency;
    let nearest_phi = (ratio * PHI).round() / PHI;
    base_frequency * nearest_phi
}

pub fn sacred_ratio(n: u32) -> f64 {
    PHI.powi(n as i32)
}

pub fn is_coherent(value: f64) -> bool {
    (0.0..=1.0).contains(&value)
}

pub fn normalize_coherence(value: f64) -> f64 {
    value.max(0.0).min(1.0)
}
