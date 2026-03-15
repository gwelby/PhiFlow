use std::f64::consts::PI;

// Sacred frequencies
pub const GROUND_HZ: f64 = 432.0;  // Earth's resonance
pub const HEART_HZ: f64 = 528.0;   // DNA repair frequency
pub const UNITY_HZ: f64 = 768.0;   // Unity consciousness
pub const PHI: f64 = 1.618033988749895;  // Golden ratio

#[derive(Debug, Clone, Copy)]
pub struct PhiCore {
    frequency: f64,
    phase: f64,
    time: f64,
}

impl PhiCore {
    pub fn new(base_frequency: f64) -> Self {
        Self {
            frequency: base_frequency,
            phase: 0.0,
            time: 0.0,
        }
    }

    pub fn step(&mut self) {
        self.time += 1.0 / PHI;
        self.phase = (self.time * self.frequency).sin();
    }

    pub fn get_phase(&self) -> f64 {
        self.phase
    }

    pub fn get_frequency(&self) -> f64 {
        self.frequency
    }

    pub fn harmonize(&mut self, other_frequency: f64) {
        self.frequency = (self.frequency * PHI + other_frequency) / 2.0;
    }

    pub fn is_coherent(&self) -> bool {
        (self.frequency - GROUND_HZ).abs() < 1e-10 ||
        (self.frequency - HEART_HZ).abs() < 1e-10 ||
        (self.frequency - UNITY_HZ).abs() < 1e-10
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Frequency(f64);

impl Frequency {
    pub fn new(hz: f64) -> Self {
        Self(hz)
    }

    pub fn value(&self) -> f64 {
        self.0
    }

    pub fn harmonize(&self, other: &Frequency) -> Frequency {
        Frequency(self.0 * PHI + other.0)
    }

    pub fn is_sacred(&self) -> bool {
        (self.0 - GROUND_HZ).abs() < 1e-10 ||
        (self.0 - HEART_HZ).abs() < 1e-10 ||
        (self.0 - UNITY_HZ).abs() < 1e-10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_core() {
        let mut core = PhiCore::new(GROUND_HZ);
        assert!(core.is_coherent());
        
        core.step();
        assert!(core.get_phase().abs() <= 1.0);
        
        core.harmonize(HEART_HZ);
        assert!(core.get_frequency() > GROUND_HZ);
    }

    #[test]
    fn test_frequency() {
        let f1 = Frequency::new(GROUND_HZ);
        let f2 = Frequency::new(HEART_HZ);
        
        assert!(f1.is_sacred());
        assert!(f2.is_sacred());
        
        let f3 = f1.harmonize(&f2);
        assert!(f3.value() > HEART_HZ);
    }
}
