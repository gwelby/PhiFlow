use super::phi_core::{Frequency, HEART_HZ, PHI};

#[derive(Debug)]
pub struct QuantumGPS {
    frequency: Frequency,
    time: f64,
}

impl QuantumGPS {
    pub fn new(frequency: Frequency) -> Self {
        Self {
            frequency,
            time: 0.0,
        }
    }

    pub fn step(&mut self) {
        self.time += 1.0 / PHI;
    }

    pub fn get_frequency(&self) -> &Frequency {
        &self.frequency
    }

    pub fn is_coherent(&self) -> bool {
        (self.time * HEART_HZ).sin().abs() > 0.9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_gps() {
        let frequency = Frequency::new(HEART_HZ);
        let mut gps = QuantumGPS::new(frequency);
        
        gps.step();
        assert!(gps.get_frequency().value() > 0.0);
    }
}
