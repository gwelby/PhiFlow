use crate::quantum::{
    PHI,
    CREATE_STATE as HEART_HZ,
    ConsciousnessField,
    quantum_consciousness::QuantumMetrics
};
use super::sacred_playground::SacredPlayground;

#[derive(Debug, Clone, Copy)]
pub struct Color {
    r: f64,
    g: f64,
    b: f64,
    a: f64,
}

impl Color {
    pub fn new(r: f64, g: f64, b: f64, a: f64) -> Self {
        Self { r, g, b, a }
    }

    pub fn golden() -> Self {
        Self::new(1.618033988749895, 1.0, 0.0, 1.0)
    }

    pub fn sacred() -> Self {
        Self::new(0.0, 1.618033988749895, 1.0, 1.0)
    }

    pub fn quantum() -> Self {
        Self::new(1.0, 0.0, 1.618033988749895, 1.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
    pub vertices: usize,
    pub coherence: f64,
    pub frequency: f64,
}

pub struct PatternVisualizer {
    playground: SacredPlayground,
    frame: usize,
    color: Color,
    phi_level: f64
}

impl PatternVisualizer {
    pub fn new() -> Self {
        Self {
            playground: SacredPlayground::new(),
            frame: 0,
            color: Color::golden(),
            phi_level: PHI
        }
    }

    pub fn step(&mut self) -> Vec<(f64, f64, f64)> {
        self.frame += 1;
        self.playground.step();
        self.playground.get_points().to_vec()
    }

    pub fn get_metrics(&self) -> FrameMetrics {
        let points = self.playground.get_points();
        FrameMetrics {
            vertices: points.len(),
            coherence: (self.frame as f64 * PHI).sin().abs(),
            frequency: HEART_HZ * PHI,
        }
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
    }

    pub fn get_color(&self) -> Color {
        self.color
    }

    pub fn generate_phi_spiral(field: &ConsciousnessField) -> Vec<(f64, f64)> {
        let metrics = field.get_quantum_metrics();
        let mut points = Vec::new();
        
        // Generate phi spiral using consciousness field
        for i in 0..144 {
            let theta = i as f64 * PHI;
            let r = metrics.coherence * (metrics.frequency / HEART_HZ).powf(PHI) * theta.exp() / PHI;
            let x = r * theta.cos();
            let y = r * theta.sin();
            points.push((x, y));
        }
        
        points
    }

    pub fn consciousness_to_frequency(consciousness: f64) -> f64 {
        HEART_HZ * consciousness.powf(PHI)
    }

    pub fn frequency_to_color(freq: f64) -> (u8, u8, u8) {
        let hue = (freq / HEART_HZ * 360.0) % 360.0;
        let (r, g, b) = Self::hsv_to_rgb(hue, 1.0, 1.0);
        ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
    }

    fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
        let c = v * s;
        let h_prime = h / 60.0;
        let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
        let m = v - c;

        let (r, g, b) = match h_prime as i32 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            5 => (c, 0.0, x),
            _ => (0.0, 0.0, 0.0)
        };

        (r + m, g + m, b + m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pattern_visualizer() {
        let mut visualizer = PatternVisualizer::new();
        
        let points = visualizer.step();
        assert!(!points.is_empty());
        
        let metrics = visualizer.get_metrics();
        assert!(metrics.vertices > 0);
        assert!(metrics.coherence >= 0.0);
        assert!(metrics.coherence <= 1.0);
        assert!(metrics.frequency > HEART_HZ);
    }

    #[test]
    fn test_colors() {
        let golden = Color::golden();
        let sacred = Color::sacred();
        let quantum = Color::quantum();
        
        assert!(golden.r > golden.g);
        assert!(sacred.g > sacred.b);
        assert!(quantum.b > quantum.r);
    }
}
