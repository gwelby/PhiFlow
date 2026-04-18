use crate::sacred::sacred_constants::*;
use serde::{Serialize, Deserialize};
use num_complex::Complex64;

// Sacred Geometry Patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacredPattern {
    name: String,
    frequency: f64,
    geometry: String,
    #[serde(with = "complex_serde")]
    amplitude: Complex64,
}

// Custom serialization for Complex64
mod complex_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(value: &Complex64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element(&value.re)?;
        tup.serialize_element(&value.im)?;
        tup.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Complex64, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::SeqAccess;
        struct ComplexVisitor;

        impl<'de> serde::de::Visitor<'de> for ComplexVisitor {
            type Value = Complex64;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a tuple of two f64 values")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let re = seq.next_element()?.unwrap_or(0.0);
                let im = seq.next_element()?.unwrap_or(0.0);
                Ok(Complex64::new(re, im))
            }
        }

        deserializer.deserialize_seq(ComplexVisitor)
    }
}

impl SacredPattern {
    pub fn new(name: &str, frequency: f64, geometry: &str) -> Self {
        Self {
            name: name.to_string(),
            frequency,
            geometry: geometry.to_string(),
            amplitude: Complex64::new(1.0, 0.0),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn geometry(&self) -> &str {
        &self.geometry
    }

    pub fn amplitude(&self) -> Complex64 {
        self.amplitude
    }

    pub fn set_amplitude(&mut self, value: Complex64) {
        self.amplitude = value;
    }
}

// Crystal Resonance Matrix
#[derive(Debug)]
pub struct CrystalMatrix {
    base_frequency: f64,
    amplification: f64,
    resonance: f64,
}

impl CrystalMatrix {
    pub fn new() -> Self {
        Self {
            base_frequency: 432.0,  // Greg's Ground State
            amplification: 1.0,
            resonance: 1.0,
        }
    }

    pub fn amplify(&mut self) -> f64 {
        self.amplification *= PHI;
        self.resonance = (self.amplification * self.base_frequency / 432.0).sin().abs();
        self.amplification
    }

    pub fn get_resonance(&self) -> f64 {
        self.resonance
    }

    pub fn tune_frequency(&mut self, freq: f64) {
        self.base_frequency = freq;
        self.resonance = (self.amplification * self.base_frequency / 432.0).sin().abs();
    }
}

// Quantum Dance Sequence
#[derive(Debug, Clone)]
pub struct DanceSequence {
    pattern: SacredPattern,
    steps: Vec<f64>,
    evolution_state: f64,
}

impl DanceSequence {
    pub fn new(pattern: SacredPattern) -> Self {
        Self {
            pattern,
            steps: Vec::new(),
            evolution_state: 1.0,
        }
    }

    pub fn add_step(&mut self, frequency: f64) {
        self.steps.push(frequency);
    }

    pub fn evolve(&mut self) {
        self.evolution_state *= PHI;
        self.pattern.set_amplitude(Complex64::new(self.evolution_state, 0.0));
    }

    pub fn verify_flow(&self) -> bool {
        self.evolution_state >= 1.0
    }

    pub fn get_pattern(&self) -> &SacredPattern {
        &self.pattern
    }
}

// Pattern Library
#[derive(Debug)]
pub struct PatternLibrary {
    sequences: Vec<DanceSequence>,
    crystal: CrystalMatrix,
}

impl PatternLibrary {
    pub fn new() -> Self {
        Self {
            sequences: Vec::new(),
            crystal: CrystalMatrix::new(),
        }
    }

    pub fn create_sequence(&mut self, pattern: SacredPattern) -> DanceSequence {
        let sequence = DanceSequence::new(pattern);
        self.sequences.push(sequence.clone());
        sequence
    }

    pub fn amplify_crystal(&mut self) -> f64 {
        self.crystal.amplify()
    }

    pub fn evolve_all(&mut self) {
        for sequence in &mut self.sequences {
            sequence.evolve();
        }
    }

    pub fn get_pattern(&self, frequency: f64) -> Option<&SacredPattern> {
        self.sequences.iter()
            .find(|s| (s.get_pattern().frequency() - frequency).abs() < 0.001)
            .map(|s| s.get_pattern())
    }

    pub fn get_metrics(&self) -> String {
        let mut metrics = String::new();
        metrics.push_str("Pattern Library Metrics:\n");
        metrics.push_str(&format!("Number of sequences: {}\n", self.sequences.len()));
        
        for (i, sequence) in self.sequences.iter().enumerate() {
            metrics.push_str(&format!("\nSequence {}:\n", i + 1));
            metrics.push_str(&format!("  Pattern: {}\n", sequence.get_pattern().name()));
            metrics.push_str(&format!("  Frequency: {:.1} Hz\n", sequence.get_pattern().frequency()));
            metrics.push_str(&format!("  Evolution: {:.3}\n", sequence.evolution_state));
        }

        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dance_sequence_evolution() {
        let pattern = SacredPattern::new("test", 432.0, "circle");
        let mut sequence = DanceSequence::new(pattern);
        
        sequence.add_step(432.0);
        sequence.add_step(528.0);
        sequence.add_step(594.0);
        
        sequence.evolve();
        assert!(sequence.verify_flow());
        
        let evolved_pattern = sequence.get_pattern();
        assert_eq!(evolved_pattern.frequency(), 432.0);
    }

    #[test]
    fn test_pattern_library_integration() {
        let mut library = PatternLibrary::new();
        let pattern = SacredPattern::new("test", 432.0, "circle");
        
        let sequence = library.create_sequence(pattern);
        assert!(sequence.verify_flow());
        
        library.evolve_all();
        let metrics = library.get_metrics();
        assert!(!metrics.is_empty());
    }
}
