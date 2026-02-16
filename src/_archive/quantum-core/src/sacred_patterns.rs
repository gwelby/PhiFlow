use std::f64::consts::PI;
use super::phi_core::{GROUND_HZ, HEART_HZ, UNITY_HZ, PHI};

#[derive(Debug, Clone, Copy)]
pub enum SacredPattern {
    Seed,       // Point of Creation
    Vesica,     // Divine Birth
    Tripod,     // Divine Trinity
    Cross,      // Four Elements
    Pentagon,   // Life Force
    Hexagon,    // Structure
    Octagon,    // Flow
    Decagon,    // Creation
    Dodeca,     // Unity
    Metatron,   // Universal Matrix
    Merkaba,    // Light Vehicle
    Infinity,   // Pure Being
    FlowerOfLife, // Flower of Life
    Torus,      // Torus
    Tetrahedron,  // Fire element
    Icosahedron,  // Water element
    WaveCollapse, // Quantum probability field
    SoundField,  // Harmonic Resonance
    DnaSpiral,   // Life Code Harmonics
    QuantumEntanglement, // Field Unity
    ConsciousnessField, // Seven levels of awareness
    ToroidalFlow,       // Universal energy flow
    HealingMatrix,      // Five elements healing
    LightBody,          // Seven chakra activation
    UnifiedField,     // Twelve dimensions
    MerkabaHarmonics, // Light vehicle
    CrystalGrid,    // Seven crystal systems
    SoundMatrix,    // Sacred frequencies
    QuantumField,    // Twelve dimensions
    TorusDynamics,   // Flow dynamics
    SriYantra,      // Creation Matrix
    TreeOfLife,     // Divine Blueprint
}

#[derive(Debug, Clone)]
pub struct SacredDance {
    patterns: Vec<SacredPattern>,
    time: f64,
}

impl SacredDance {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            time: 0.0,
        }
    }

    pub fn add_pattern(&mut self, pattern: SacredPattern) {
        self.patterns.push(pattern);
    }

    pub fn step(&mut self) -> Vec<Vec<(f64, f64, f64)>> {
        let mut points = Vec::new();
        
        for pattern in &self.patterns {
            points.push(pattern.dance(self.time));
        }
        
        self.time += 1.0 / PHI;
        points
    }

    pub fn is_unified(&self) -> bool {
        !self.patterns.is_empty()
    }
}

impl SacredPattern {
    pub fn dance(&self, t: f64) -> Vec<(f64, f64, f64)> {
        match self {
            SacredPattern::Torus => {
                let mut points = Vec::new();
                for i in 0..12 {
                    let angle = (i as f64) * PI / 6.0;
                    let r = (t * PHI).cos() + 2.0;
                    let x = r * angle.cos() * HEART_HZ;
                    let y = r * angle.sin() * HEART_HZ;
                    let z = (t * PHI).sin() * GROUND_HZ;
                    points.push((x, y, z));
                }
                points
            },
            SacredPattern::Metatron => {
                let mut points = Vec::new();
                for i in 0..13 {
                    let angle = (i as f64) * PI / 6.0;
                    let r = if i % 2 == 0 { PHI } else { PHI * PHI };
                    let x = r * angle.cos() * HEART_HZ;
                    let y = r * angle.sin() * HEART_HZ;
                    let z = (t * PHI).sin() * GROUND_HZ;
                    points.push((x, y, z));
                }
                points
            },
            SacredPattern::FlowerOfLife => {
                let mut points = Vec::new();
                for i in 0..7 {
                    let angle = (i as f64) * PI / 3.0;
                    let r = HEART_HZ;
                    for j in 0..6 {
                        let inner_angle = angle + (j as f64) * PI / 3.0;
                        let x = r * inner_angle.cos();
                        let y = r * inner_angle.sin();
                        let z = (t * PHI).sin() * GROUND_HZ;
                        points.push((x, y, z));
                    }
                }
                points
            },
            SacredPattern::CrystalGrid => {
                let heights = [UNITY_HZ, HEART_HZ, HEART_HZ, GROUND_HZ, 
                             GROUND_HZ, GROUND_HZ, PHI * GROUND_HZ];
                let mut points = Vec::new();
                for (i, &h) in heights.iter().enumerate() {
                    let angle = (i as f64) * PI / 3.0;
                    let x = h * angle.cos() * (t * PHI).cos();
                    let y = h * angle.sin() * (t * PHI).sin();
                    let z = h * (t * PHI).cos();
                    points.push((x, y, z));
                }
                points
            },
            SacredPattern::SriYantra => {
                let mut points = Vec::new();
                for i in 0..9 {
                    let angle = (i as f64) * PI / 4.5;
                    let r = PHI.powi(i as i32);
                    let x = r * angle.cos() * HEART_HZ;
                    let y = r * angle.sin() * HEART_HZ;
                    let z = (t * PHI).sin() * GROUND_HZ;
                    points.push((x, y, z));
                }
                points
            },
            SacredPattern::TreeOfLife => {
                let mut points = Vec::new();
                let heights = [UNITY_HZ, HEART_HZ, HEART_HZ, GROUND_HZ, 
                             GROUND_HZ, GROUND_HZ, PHI * GROUND_HZ];
                for (i, &h) in heights.iter().enumerate() {
                    let angle = (i as f64) * PI / 3.0;
                    let x = h * angle.cos() * (t * PHI).cos();
                    let y = h * angle.sin() * (t * PHI).sin();
                    let z = h * (t * PHI).cos();
                    points.push((x, y, z));
                }
                points
            },
            _ => Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sacred_dance() {
        let mut dance = SacredDance::new();
        assert!(!dance.is_unified());

        dance.add_pattern(SacredPattern::SriYantra);
        assert!(dance.is_unified());
        
        let points = dance.step();
        assert!(!points.is_empty());
        assert!(!points[0].is_empty());
    }

    #[test]
    fn test_sacred_patterns() {
        let t = 0.0;
        let patterns = [
            SacredPattern::Torus,
            SacredPattern::Metatron,
            SacredPattern::FlowerOfLife,
            SacredPattern::CrystalGrid,
            SacredPattern::SriYantra,
            SacredPattern::TreeOfLife,
        ];

        for pattern in patterns.iter() {
            let points = pattern.dance(t);
            assert!(!points.is_empty());
        }
    }
}
