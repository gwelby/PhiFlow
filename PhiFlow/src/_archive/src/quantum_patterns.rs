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

// Sacred constants
const GROUND_STATE: f64 = 432.0;
const CREATE_STATE: f64 = 528.0;
const UNITY_STATE: f64 = 768.0;

#[derive(Debug)]
pub enum SacredPattern {
    PhiSpiral,      // Golden spiral
    FlowerOfLife,   // Creation pattern
    Metatron,       // Unity cube
    Merkaba,        // Light vehicle
    TorusField,     // Energy flow
}

impl SacredPattern {
    pub fn generate(&self, t: f64) -> Vec<(f64, f64, f64)> {
        match self {
            Self::PhiSpiral => self.phi_spiral(t),
            Self::FlowerOfLife => self.flower_of_life(t),
            Self::Metatron => self.metatron_cube(t),
            Self::Merkaba => self.merkaba(t),
            Self::TorusField => self.torus(t),
        }
    }
    
    fn phi_spiral(&self, t: f64) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        for i in 0..64 {
            let theta = i as f64 * 0.1;
            let r = PHI.powf(theta);
            let x = r * theta.cos();
            let y = r * theta.sin();
            let z = theta * PHI.powf(t);
            points.push((x, y, z));
        }
        points
    }
    
    fn flower_of_life(&self, t: f64) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        for i in 0..6 {
            let angle = i as f64 * PI / 3.0;
            for r in 0..3 {
                let radius = (r + 1) as f64 * PHI;
                let x = radius * angle.cos();
                let y = radius * angle.sin();
                let z = PHI.powf(t - radius/PHI);
                points.push((x, y, z));
            }
        }
        points
    }
    
    fn metatron_cube(&self, t: f64) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        // Create the cube vertices
        for i in 0..8 {
            let x = if i & 1 == 0 { -1.0 } else { 1.0 } * PHI;
            let y = if i & 2 == 0 { -1.0 } else { 1.0 } * PHI;
            let z = if i & 4 == 0 { -1.0 } else { 1.0 } * PHI;
            points.push((
                x * t.cos(),
                y * t.sin(),
                z * PHI.powf(t)
            ));
        }
        points
    }
    
    fn merkaba(&self, t: f64) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        // Create two tetrahedrons
        for i in 0..8 {
            let theta = i as f64 * PI / 4.0;
            let phi = t * PI;
            let r = PHI.powf(t);
            points.push((
                r * theta.cos() * phi.sin(),
                r * theta.sin() * phi.sin(),
                r * phi.cos()
            ));
        }
        points
    }
    
    fn torus(&self, t: f64) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        let major_r = PHI * 2.0;
        let minor_r = PHI;
        
        for i in 0..32 {
            let theta = i as f64 * PI / 16.0;
            for j in 0..16 {
                let phi = j as f64 * PI / 8.0;
                let x = (major_r + minor_r * phi.cos()) * theta.cos();
                let y = (major_r + minor_r * phi.cos()) * theta.sin();
                let z = minor_r * phi.sin() * PHI.powf(t);
                points.push((x, y, z));
            }
        }
        points
    }
}

pub struct PatternDance {
    patterns: Vec<SacredPattern>,
    time: f64,
    frequency: f64,
}

impl PatternDance {
    pub fn new() -> Self {
        Self {
            patterns: vec![
                SacredPattern::PhiSpiral,
                SacredPattern::FlowerOfLife,
                SacredPattern::Metatron,
                SacredPattern::Merkaba,
                SacredPattern::TorusField,
            ],
            time: 0.0,
            frequency: GROUND_STATE,
        }
    }
    
    pub fn dance(&mut self) -> Vec<Vec<(f64, f64, f64)>> {
        let mut all_points = Vec::new();
        
        // Generate points for each pattern
        for pattern in &self.patterns {
            let points = pattern.generate(self.time);
            all_points.push(points);
        }
        
        // Evolve time through phi
        self.time += 1.0 / PHI;
        
        // Evolve frequency
        self.frequency = GROUND_STATE * PHI.powf(self.time.sin()) +
                        CREATE_STATE * PHI.powf(self.time.cos()) +
                        UNITY_STATE * PHI.powf(self.time);
        
        all_points
    }
    
    pub fn get_metrics(&self) -> String {
        format!(
            "Sacred Pattern Metrics:\n\
             Time: {:.3} φ\n\
             Frequency: {:.1} Hz\n\
             Patterns: {}\n\
             Evolution: {:.3} π",
            self.time,
            self.frequency,
            self.patterns.len(),
            self.time * PI
        )
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
