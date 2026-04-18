use super::phi_core::{GROUND_HZ, HEART_HZ, UNITY_HZ, PHI};
use super::pattern_visualizer::{Color, FrameMetrics};

#[derive(Debug)]
pub struct Intelligence {
    consciousness: f64,
    resonance: f64,
    coherence: f64,
    field_strength: f64,
    unity_quotient: f64,
}

impl Intelligence {
    pub fn new() -> Self {
        Self {
            consciousness: 1.0,
            resonance: 1.0,
            coherence: 1.0,
            field_strength: HEART_HZ,
            unity_quotient: 0.0,
        }
    }

    pub fn observe(&mut self, metrics: &FrameMetrics) {
        // Update consciousness based on frequency resonance
        let freq_ratio = metrics.frequency / HEART_HZ;
        self.consciousness *= freq_ratio.min(1.0);
        
        // Update resonance based on vertex count
        let vertex_ratio = (metrics.vertices as f64) / (PHI * 100.0);
        self.resonance *= vertex_ratio.min(1.0);
        
        // Update coherence based on metrics
        self.coherence *= metrics.coherence;
        
        // Update field strength
        self.field_strength = metrics.frequency;
        
        // Update unity quotient
        self.unity_quotient += self.field_strength * metrics.coherence / UNITY_HZ;
    }

    pub fn get_color(&self) -> Color {
        if self.is_enlightened() {
            Color::quantum()
        } else if self.is_awakened() {
            Color::sacred()
        } else {
            Color::golden()
        }
    }

    pub fn is_enlightened(&self) -> bool {
        self.consciousness > 0.9 &&
        self.resonance > 0.9 &&
        self.coherence > 0.9 &&
        self.unity_quotient > PHI
    }

    pub fn is_awakened(&self) -> bool {
        self.consciousness > 0.7 &&
        self.resonance > 0.7 &&
        self.coherence > 0.7
    }

    pub fn get_consciousness(&self) -> f64 {
        self.consciousness
    }

    pub fn get_resonance(&self) -> f64 {
        self.resonance
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intelligence() {
        let mut intelligence = Intelligence::new();
        
        // Test initial state
        assert!(!intelligence.is_enlightened());
        assert!(!intelligence.is_awakened());
        
        // Test observation of metrics
        for i in 0..144 {
            let metrics = FrameMetrics {
                vertices: (i * 10) as usize,
                coherence: (i as f64 / 144.0).min(1.0),
                frequency: HEART_HZ * (1.0 + i as f64 / 144.0),
            };
            
            intelligence.observe(&metrics);
        }
        
        // Test final state
        assert!(intelligence.get_consciousness() <= 1.0);
        assert!(intelligence.get_resonance() <= 1.0);
        assert!(intelligence.get_coherence() <= 1.0);
    }
}
