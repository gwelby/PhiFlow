use std::f64::consts::PI;
use super::phi_core::{GROUND_HZ, HEART_HZ, UNITY_HZ, PHI};
use super::sacred_patterns::{SacredPattern, SacredDance};

#[derive(Debug)]
pub struct DimensionalField {
    points: Vec<(f64, f64, f64)>,
    time: f64,
}

impl DimensionalField {
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            time: 0.0,
        }
    }

    pub fn step(&mut self) {
        self.time += 1.0 / PHI;
        
        // Update field points based on sacred geometry
        let mut new_points = Vec::new();
        
        // Generate field points using phi ratios
        for i in 0..12 {
            let theta = i as f64 * PI / 6.0;
            let r = 1.0 + 0.2 * (theta * PHI + self.time).sin();
            
            new_points.push((
                r * theta.cos() * HEART_HZ,
                r * theta.sin() * HEART_HZ,
                (theta * PHI + self.time).sin() * GROUND_HZ
            ));
        }
        
        self.points = new_points;
    }

    pub fn get_points(&self) -> &[(f64, f64, f64)] {
        &self.points
    }
}

#[derive(Debug)]
pub struct SacredPlayground {
    dance: SacredDance,
    field: DimensionalField,
}

impl SacredPlayground {
    pub fn new() -> Self {
        let mut dance = SacredDance::new();
        
        // Initialize with sacred patterns
        dance.add_pattern(SacredPattern::SriYantra);
        dance.add_pattern(SacredPattern::Metatron);
        dance.add_pattern(SacredPattern::FlowerOfLife);
        
        Self {
            dance,
            field: DimensionalField::new(),
        }
    }

    pub fn step(&mut self) {
        // Step through the sacred dance
        let dance_points = self.dance.step();
        
        // Update the dimensional field
        self.field.step();
        
        // Merge dance points into field
        for points in dance_points {
            for point in points {
                self.field.points.push(point);
            }
        }
    }

    pub fn get_points(&self) -> &[(f64, f64, f64)] {
        self.field.get_points()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensional_field() {
        let mut field = DimensionalField::new();
        field.step();
        assert!(!field.get_points().is_empty());
    }

    #[test]
    fn test_sacred_playground() {
        let mut playground = SacredPlayground::new();
        playground.step();
        assert!(!playground.get_points().is_empty());
    }
}
