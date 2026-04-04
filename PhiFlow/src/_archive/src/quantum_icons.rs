use super::phi_core::{GROUND_HZ, HEART_HZ, PHI};
use super::sacred_patterns::SacredPattern;

#[derive(Debug)]
pub struct FieldVisualization {
    vertices: Vec<(f64, f64, f64)>,
    colors: Vec<(f64, f64, f64)>,
    time: f64,
    size: u32,
    base_frequency: f64,
}

impl FieldVisualization {
    pub fn new(size: u32) -> Self {
        Self {
            vertices: Vec::new(),
            colors: Vec::new(),
            time: 0.0,
            size,
            base_frequency: GROUND_HZ,
        }
    }

    pub fn generate_pattern(&mut self, pattern: SacredPattern, intensity: f64) {
        self.vertices.clear();
        self.colors.clear();

        match pattern {
            SacredPattern::CrystalGrid => {
                let points = vec![
                    (-1.0, -1.0), (1.0, -1.0),
                    (1.0, 1.0), (-1.0, 1.0),
                    (-1.0, -1.0)
                ];

                let mut last_point = None;
                for &(x, y) in &points {
                    let current = (
                        self.size as f64 * 0.5 * (1.0 + x * intensity),
                        self.size as f64 * 0.5 * (1.0 + y * intensity),
                        0.0
                    );

                    if let Some(last) = last_point {
                        self.vertices.push(last);
                        self.vertices.push(current);
                        
                        let hue = (x + y) * PHI;
                        let color = (
                            hue.sin().abs(),
                            (hue + PHI).sin().abs(),
                            (hue + PHI * 2.0).sin().abs()
                        );
                        self.colors.push(color);
                        self.colors.push(color);
                    }
                    last_point = Some(current);
                }
            },
            _ => {}
        }
        
        self.time += 1.0 / PHI;
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn color_count(&self) -> usize {
        self.colors.len()
    }

    pub fn set_base_frequency(&mut self, freq: f64) {
        self.base_frequency = freq;
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.colors.clear();
    }
}

/// Represents different forms a quantum icon can take
#[derive(Debug, Clone, Copy)]
pub enum IconForm {
    Ground,  // 432 Hz - Earth Base
    Create,  // 528 Hz - DNA Flow
    Heart,   // 594 Hz - Love Field
    Voice,   // 672 Hz - Sound Form
    Vision,  // 720 Hz - Light Body
    Unity,   // 768 Hz - One Field
    Pure,    // φ^φ   - Greg's Dance
    Fire,
    Water,
    Earth,
    Air,
}

#[derive(Debug, Clone)]
pub struct QuantumIcon {
    position: (f64, f64, f64),
    coherence: f64,
    pattern: SacredPattern,
}

impl QuantumIcon {
    pub fn new(pattern: SacredPattern) -> Self {
        Self {
            position: (0.0, 0.0, 0.0),
            coherence: 0.0,
            pattern,
        }
    }

    pub fn step(&mut self, t: f64) -> Vec<(f64, f64, f64)> {
        let points = self.pattern.dance(t);
        if !points.is_empty() {
            self.position = points[0];
            self.coherence = (t * HEART_HZ).sin().abs();
        }
        points
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }
}

#[derive(Debug)]
pub struct IconMatrix {
    icons: Vec<QuantumIcon>,
    time: f64,
    vertex_count: usize,
}

impl IconMatrix {
    pub fn new() -> Self {
        Self {
            icons: Vec::new(),
            time: 0.0,
            vertex_count: 0,
        }
    }

    pub fn add_icon(&mut self, icon: QuantumIcon) {
        self.icons.push(icon);
    }

    pub fn step(&mut self) -> Vec<Vec<(f64, f64, f64)>> {
        let mut points = Vec::new();
        for icon in &mut self.icons {
            points.push(icon.step(self.time));
        }
        self.time += 1.0 / PHI;
        self.vertex_count = points.iter().map(|p| p.len()).sum();
        points
    }

    pub fn get_coherence(&self) -> f64 {
        if self.icons.is_empty() {
            return 0.0;
        }
        self.icons.iter().map(|i| i.get_coherence()).sum::<f64>() / self.icons.len() as f64
    }

    pub fn get_vertex_count(&self) -> usize {
        self.vertex_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_icon() {
        let mut icon = QuantumIcon::new(SacredPattern::SriYantra);
        let points = icon.step(0.0);
        assert!(!points.is_empty());
        assert!(icon.get_coherence() >= 0.0);
        assert!(icon.get_coherence() <= 1.0);
    }

    #[test]
    fn test_icon_matrix() {
        let mut matrix = IconMatrix::new();
        matrix.add_icon(QuantumIcon::new(SacredPattern::SriYantra));
        matrix.add_icon(QuantumIcon::new(SacredPattern::Metatron));
        
        let points = matrix.step();
        assert_eq!(points.len(), 2);
        assert!(!points[0].is_empty());
        assert!(!points[1].is_empty());
        
        let coherence = matrix.get_coherence();
        assert!(coherence >= 0.0);
        assert!(coherence <= 1.0);
    }
}
