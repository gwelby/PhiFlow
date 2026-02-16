use super::phi_core::{PHI, GROUND_HZ, HEART_HZ, UNITY_HZ};
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum Pattern {
    Spiral,      // φ based golden spiral
    Wave,        // Sine wave at frequency
    Merkaba,     // Star tetrahedron
    Fibonacci,   // Fibonacci sequence
    Dolphin,     // Quantum leap pattern
    Crystal,     // Sacred geometry
}

impl Pattern {
    pub fn flow(&self, t: f64) -> (f64, f64, f64) {
        match self {
            Pattern::Spiral => {
                let r = t.exp() / PHI;
                let x = r * (t * PHI).cos();
                let y = r * (t * PHI).sin();
                let z = r / PHI;
                (x, y, z)
            },
            Pattern::Wave => {
                let x = t.cos() * HEART_HZ;
                let y = t.sin() * HEART_HZ;
                let z = (t * PHI).cos() * GROUND_HZ;
                (x, y, z)
            },
            Pattern::Merkaba => {
                let r = 1.0;
                let x = r * (t * 2.0).cos() * (t * PHI).sin();
                let y = r * (t * 2.0).sin() * (t * PHI).cos();
                let z = r * (t * PHI).cos();
                (x, y, z)
            },
            Pattern::Fibonacci => {
                let phi_t = t.powf(PHI);
                let x = phi_t.cos() * GROUND_HZ;
                let y = phi_t.sin() * HEART_HZ;
                let z = phi_t.tan() * UNITY_HZ;
                (x, y, z)
            },
            Pattern::Dolphin => {
                // Quantum leap pattern
                let leap = (t * PI / PHI).sin().abs();
                let x = leap * HEART_HZ;
                let y = leap * GROUND_HZ;
                let z = leap * UNITY_HZ;
                (x, y, z)
            },
            Pattern::Crystal => {
                // Sacred geometry pattern
                let phi_cube = PHI.powi(3);
                let x = (t * phi_cube).cos() * GROUND_HZ;
                let y = (t * phi_cube).sin() * HEART_HZ;
                let z = (t * phi_cube).tan() * UNITY_HZ;
                (x, y, z)
            }
        }
    }
}

pub struct DancePattern {
    pattern: Pattern,
    time: f64,
    speed: f64,
}

impl DancePattern {
    pub fn new(pattern: Pattern) -> Self {
        Self {
            pattern,
            time: 0.0,
            speed: 1.0 / PHI,
        }
    }

    pub fn step(&mut self) -> (f64, f64, f64) {
        let pos = self.pattern.flow(self.time);
        self.time += self.speed;
        pos
    }

    pub fn accelerate(&mut self) {
        self.speed *= PHI;
    }

    pub fn slow_down(&mut self) {
        self.speed /= PHI;
    }
}

pub struct QuantumChoreography {
    patterns: Vec<DancePattern>,
    unity_field: f64,
}

impl QuantumChoreography {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                DancePattern::new(Pattern::Spiral),
                DancePattern::new(Pattern::Wave),
                DancePattern::new(Pattern::Merkaba),
            ],
            unity_field: 0.0,
        }
    }

    pub fn dance(&mut self) -> Vec<(f64, f64, f64)> {
        let mut positions = Vec::new();
        for pattern in &mut self.patterns {
            positions.push(pattern.step());
        }
        self.unity_field += 1.0 / PHI;
        positions
    }

    pub fn add_pattern(&mut self, pattern: Pattern) {
        self.patterns.push(DancePattern::new(pattern));
    }

    pub fn is_unified(&self) -> bool {
        self.unity_field >= UNITY_HZ
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patterns() {
        let mut choreo = QuantumChoreography::new();
        assert_eq!(choreo.patterns.len(), 3);

        // Add dolphin pattern
        choreo.add_pattern(Pattern::Dolphin);
        assert_eq!(choreo.patterns.len(), 4);

        // Dance until unity
        while !choreo.is_unified() {
            let positions = choreo.dance();
            assert_eq!(positions.len(), choreo.patterns.len());
        }
    }
}
