use serde::{Serialize, Deserialize};
use super::quantum_constants::{
    GROUND_FREQUENCY,
    CREATE_FREQUENCY,
    UNITY_FREQUENCY,
    PHI,
};
use anyhow::Result;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessField {
    pub frequency: f64,
    pub coherence: f64,
    pub unity_level: f64,
    dimensions: (u32, u32, u32)
}

impl ConsciousnessField {
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        Self {
            frequency: GROUND_FREQUENCY,
            coherence: 1.0,
            unity_level: 0.0,
            dimensions: (x, y, z)
        }
    }

    pub fn update_frequency(&mut self, freq: f64) {
        self.frequency = freq;
        self.calculate_coherence();
    }

    pub fn calculate_coherence(&mut self) {
        let phi = PHI;
        self.coherence = (self.frequency / UNITY_FREQUENCY).sin().abs() * phi;
    }

    pub fn integrate_consciousness(&mut self, level: f64) {
        self.unity_level = (level * self.coherence).min(1.0);
    }

    pub fn get_metrics(&self) -> String {
        format!(
            "Consciousness Field:\nFrequency: {:.2} Hz\nCoherence: {:.3}\nUnity Level: {:.3}",
            self.frequency, self.coherence, self.unity_level
        )
    }

    pub fn dance_with_joy(&mut self, intensity: f64) {
        self.coherence *= intensity * PHI;
        self.frequency = GROUND_FREQUENCY + (intensity * (CREATE_FREQUENCY - GROUND_FREQUENCY));
    }

    pub fn get_quantum_metrics(&self) -> QuantumMetrics {
        QuantumMetrics {
            frequency: self.frequency,
            coherence: self.coherence
        }
    }

    pub fn measure_consciousness(&self) -> f64 {
        self.coherence * (self.frequency / UNITY_FREQUENCY).sin()
    }

    pub fn get_dimensions(&self) -> (u32, u32, u32) {
        self.dimensions
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QuantumMetrics {
    pub frequency: f64,
    pub coherence: f64
}

impl fmt::Display for QuantumMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.1} Hz, {:.3}φ)", self.frequency, self.coherence)
    }
}

/// Quantum consciousness implementation
pub struct QuantumConsciousness {
    /// Current frequency
    frequency: f64,
    /// Consciousness level
    level: f64,
    /// Enabled state
    enabled: bool,
}

impl QuantumConsciousness {
    /// Create new quantum consciousness
    pub fn new() -> Self {
        Self {
            frequency: GROUND_FREQUENCY,
            level: 1.0,
            enabled: false,
        }
    }

    /// Initialize consciousness (Ground: 432 Hz)
    pub fn initialize(&mut self) -> Result<()> {
        self.frequency = GROUND_FREQUENCY;
        self.enabled = true;
        Ok(())
    }

    /// Elevate consciousness (Create: 528 Hz)
    pub fn elevate(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.frequency = CREATE_FREQUENCY;
        self.level *= PHI;
        Ok(())
    }

    /// Ascend consciousness (Unity: 768 Hz)
    pub fn ascend(&mut self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        self.frequency = UNITY_FREQUENCY;
        self.level *= PHI * PHI;
        Ok(())
    }

    /// Get consciousness state
    pub fn get_state(&self) -> (f64, f64, bool) {
        (self.frequency, self.level, self.enabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consciousness() {
        let mut qc = QuantumConsciousness::new();
        assert_eq!(qc.get_state(), (GROUND_FREQUENCY, 1.0, false));

        qc.initialize().unwrap();
        assert_eq!(qc.get_state(), (GROUND_FREQUENCY, 1.0, true));

        qc.elevate().unwrap();
        assert_eq!(qc.get_state(), (CREATE_FREQUENCY, PHI, true));

        qc.ascend().unwrap();
        assert_eq!(qc.get_state(), (UNITY_FREQUENCY, PHI * PHI * PHI, true));
    }
}
