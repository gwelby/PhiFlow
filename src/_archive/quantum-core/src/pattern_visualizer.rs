use std::f64::consts::PI;
use super::phi_core::{GROUND_HZ, HEART_HZ, UNITY_HZ, PHI};
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
}

impl PatternVisualizer {
    pub fn new() -> Self {
        Self {
            playground: SacredPlayground::new(),
            frame: 0,
            color: Color::golden(),
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
