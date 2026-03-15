// PhiFlow Quantum Types - Common types for quantum operations

use serde::{Serialize, Deserialize};

// Re-export the main types from mod.rs for convenience
pub use super::{
    QuantumBackend, QuantumConfig, QuantumCircuit, QuantumGate,
    QuantumResult, QuantumCapabilities, BackendStatus, QuantumError,
    QuantumResult2
};

// Sacred frequency constants used in quantum operations
pub const SACRED_FREQUENCIES: &[u32] = &[432, 528, 594, 639, 693, 741, 852, 963];
pub const PHI: f64 = 1.618033988749895;

// Phi-harmonic quantum constants
pub const PHI_SQUARED: f64 = PHI * PHI;
pub const PHI_CUBED: f64 = PHI * PHI * PHI;
pub const LAMBDA: f64 = 1.0 / PHI; // Divine complement

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacredFrequencyOperation {
    pub frequency: u32,
    pub qubits: Vec<u32>,
    pub duration: f64,
    pub consciousness_coupling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiHarmonicOperation {
    pub qubit: u32,
    pub phi_power: f64,
    pub axis: PhiAxis,
    pub coupling_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhiAxis {
    X,
    Y,
    Z,
    Spherical,
}

#[derive(Debug, Clone)]
pub struct ConsciousnessQuantumCoupling {
    pub coherence_threshold: f64,
    pub sacred_frequency_lock: Option<u32>,
    pub phi_resonance: f64,
    pub quantum_authorization: bool,
}

impl Default for ConsciousnessQuantumCoupling {
    fn default() -> Self {
        ConsciousnessQuantumCoupling {
            coherence_threshold: 0.8,
            sacred_frequency_lock: None,
            phi_resonance: 0.0,
            quantum_authorization: false,
        }
    }
}

// Utility functions for quantum operations
pub fn frequency_to_quantum_angle(frequency: u32) -> f64 {
    // Convert sacred frequency to quantum rotation angle
    let base_freq = 432.0; // Base frequency
    let ratio = frequency as f64 / base_freq;
    ratio * std::f64::consts::PI / 2.0
}

pub fn phi_power_to_angle(phi_power: f64) -> f64 {
    // Convert phi power to quantum rotation angle
    phi_power * std::f64::consts::PI / PHI
}

pub fn calculate_phi_resonance(frequency: u32) -> f64 {
    // Calculate how well frequency resonates with phi
    let phi_frequency = 432.0 * PHI;
    let diff = (frequency as f64 - phi_frequency).abs();
    1.0 / (1.0 + diff / 100.0)
}

pub fn is_sacred_frequency(frequency: u32) -> bool {
    SACRED_FREQUENCIES.contains(&frequency)
}

pub fn get_nearest_sacred_frequency(frequency: u32) -> u32 {
    *SACRED_FREQUENCIES
        .iter()
        .min_by_key(|&&f| (f as i32 - frequency as i32).abs())
        .unwrap_or(&432)
}