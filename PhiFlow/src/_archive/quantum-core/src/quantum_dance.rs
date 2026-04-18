use super::phi_core::{Frequency, PHI};
use super::sacred_patterns::{SacredPattern, SacredDance};

#[derive(Debug)]
pub struct QuantumDance {
    dance: SacredDance,
    frequency: Frequency,
    time: f64,
}

impl QuantumDance {
    pub fn new(frequency: Frequency) -> Self {
        let mut dance = SacredDance::new();
        dance.add_pattern(SacredPattern::SriYantra);
        dance.add_pattern(SacredPattern::Metatron);
        dance.add_pattern(SacredPattern::FlowerOfLife);
        
        Self {
            dance,
            frequency,
            time: 0.0,
        }
    }

    pub fn step(&mut self) -> Vec<Vec<(f64, f64, f64)>> {
        let points = self.dance.step();
        self.time += 1.0 / PHI;
        points
    }

    pub fn is_unified(&self) -> bool {
        self.dance.is_unified()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi_core::HEART_HZ;

    #[test]
    fn test_quantum_dance() {
        let frequency = Frequency::new(HEART_HZ);
        let mut dance = QuantumDance::new(frequency);
        
        assert!(dance.is_unified());
        let points = dance.step();
        assert!(!points.is_empty());
    }
}
