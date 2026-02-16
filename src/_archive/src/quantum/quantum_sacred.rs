use serde::{Serialize, Deserialize};
use num_complex::Complex64;
use anyhow::Result;
use super::quantum_constants::{
    GROUND_FREQUENCY,
    CREATE_FREQUENCY,
    UNITY_FREQUENCY,
    PHI,
    PHI_SQUARED,
    PHI_CUBED,
    HUMAN_SCALE,
    SacredPattern,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SacredGeometry {
    name: String,
    frequency: f64,
    #[serde(with = "complex_serde")]
    vertices: Vec<Complex64>,
    dimensions: u8,
    resonance: f64,
    pattern: SacredPattern,
    coherence: f64,
}

// Custom serialization for Complex64
mod complex_serde {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(vertices: &Vec<Complex64>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        let mut seq = serializer.serialize_seq(Some(vertices.len() * 2))?;
        for v in vertices {
            seq.serialize_element(&v.re)?;
            seq.serialize_element(&v.im)?;
        }
        seq.end()
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Complex64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::SeqAccess;
        struct ComplexVisitor;

        impl<'de> serde::de::Visitor<'de> for ComplexVisitor {
            type Value = Vec<Complex64>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sequence of real and imaginary parts")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut vertices = Vec::new();
                while let Some(re) = seq.next_element()? {
                    let im = seq.next_element()?.unwrap_or(0.0);
                    vertices.push(Complex64::new(re, im));
                }
                Ok(vertices)
            }
        }

        deserializer.deserialize_seq(ComplexVisitor)
    }
}

impl SacredGeometry {
    pub fn new(name: &str, frequency: f64) -> Result<Self> {
        let mut geometry = Self {
            name: name.to_string(),
            frequency,
            vertices: Vec::new(),
            dimensions: 3,
            resonance: PHI,
            pattern: SacredPattern::Spiral,
            coherence: HUMAN_SCALE,
        };
        
        geometry.vertices = geometry.calculate_vertices(name);
        geometry.resonance = geometry.calculate_resonance();
        
        Ok(geometry)
    }

    fn calculate_vertices(&self, shape: &str) -> Vec<Complex64> {
        match shape.to_lowercase().as_str() {
            "metatron" => {
                // Metatron's Cube - 13 circles
                let mut vertices = Vec::with_capacity(13);
                for i in 0..13 {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / 13.0;
                    vertices.push(Complex64::new(
                        PHI * angle.cos(),
                        PHI * angle.sin()
                    ));
                }
                vertices
            },
            "flower" => {
                // Flower of Life - 19 circles
                let mut vertices = Vec::with_capacity(19);
                for i in 0..19 {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / 19.0;
                    vertices.push(Complex64::new(
                        PHI_SQUARED * angle.cos(),
                        PHI_SQUARED * angle.sin()
                    ));
                }
                vertices
            },
            "merkaba" => {
                // Merkaba - Two tetrahedra (8 vertices)
                let mut vertices = Vec::with_capacity(8);
                for i in 0..8 {
                    let angle = 2.0 * std::f64::consts::PI * (i as f64) / 8.0;
                    vertices.push(Complex64::new(
                        PHI_CUBED * angle.cos(),
                        PHI_CUBED * angle.sin()
                    ));
                }
                vertices
            },
            _ => {
                // Default to Phi Spiral
                let mut vertices = Vec::with_capacity(21);
                for i in 0..21 {
                    let angle = PHI * (i as f64);
                    let r = PHI.powf(i as f64 / 7.0);
                    vertices.push(Complex64::new(
                        r * angle.cos(),
                        r * angle.sin()
                    ));
                }
                vertices
            }
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn frequency(&self) -> f64 {
        self.frequency
    }

    pub fn vertices(&self) -> &[Complex64] {
        &self.vertices
    }

    pub fn coherence(&self) -> f64 {
        // Calculate coherence based on frequency ratios
        let ground_ratio = self.frequency / GROUND_FREQUENCY;
        let create_ratio = self.frequency / CREATE_FREQUENCY;
        let unity_ratio = self.frequency / UNITY_FREQUENCY;
        
        (ground_ratio * create_ratio * unity_ratio).powf(1.0/3.0)
    }

    pub fn initialize_field(&mut self, ground_freq: f64, create_freq: f64) -> Result<()> {
        self.frequency = ground_freq;
        self.resonance = create_freq / ground_freq * PHI;
        self.coherence = self.coherence();
        Ok(())
    }

    pub fn calculate_resonance(&self) -> f64 {
        self.vertices.iter()
            .map(|v| v.norm())
            .sum::<f64>() / self.vertices.len() as f64
    }

    pub fn initialize_field_default(&self) -> Result<f64> {
        Ok(GROUND_FREQUENCY * PHI)
    }

    pub fn calculate_resonance_default(&self) -> f64 {
        PHI * PHI * PHI // φ³
    }

    pub fn dance_sacred_pattern(&mut self, pattern: SacredPattern) -> Result<()> {
        self.pattern = pattern;
        self.frequency = match pattern {
            SacredPattern::Ground => GROUND_FREQUENCY,
            SacredPattern::Create => CREATE_FREQUENCY,
            SacredPattern::Unity => UNITY_FREQUENCY,
            _ => self.frequency * PHI,
        };
        self.coherence = self.coherence();
        Ok(())
    }

    pub fn get_pattern_symbol(&self) -> &'static str {
        self.pattern.symbol()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sacred_geometry() -> Result<()> {
        let geometry = SacredGeometry::new("Metatron", 528.0)?;
        assert_eq!(geometry.name(), "Metatron");
        assert_eq!(geometry.frequency(), 528.0);
        assert!(geometry.coherence() > 0.0);
        Ok(())
    }
}
