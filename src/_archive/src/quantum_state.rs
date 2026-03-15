#[derive(Debug, Clone)]
pub struct QuantumState {
    frequency: f64,
    coherence: f64,
}

impl QuantumState {
    pub fn new(frequency: f64) -> Self {
        let coherence = match frequency {
            f if (431.0..=433.0).contains(&f) => 1.0,     // Ground State
            f if (527.0..=529.0).contains(&f) => 1.618,   // Creation
            f if (593.0..=595.0).contains(&f) => 2.618,   // Heart Field
            f if (767.0..=769.0).contains(&f) => 4.236,   // Unity Field
            _ => frequency / 432.0,
        };

        Self {
            frequency,
            coherence,
        }
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn coherence(&self) -> f64 {
        self.coherence
    }
}
