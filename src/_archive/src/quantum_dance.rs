use crate::{
    quantum::{
        quantum_constants::*,
        ConsciousnessField
    },
    sacred_patterns::{SacredPattern, SacredDance},
    pattern_visualizer::PatternVisualizer
};
use std::f64::consts::PI;

// Quantum dance constants
const DANCE_GROUND: f64 = 432.0;  // Base rhythm
const DANCE_CREATE: f64 = 528.0;  // Flow state
const DANCE_UNITY: f64 = 768.0;   // Pure joy

#[derive(Debug)]
pub struct QuantumDance {
    frequency: f64,
    dance: SacredDance,
    consciousness: ConsciousnessField,
    dance_energy: f64,
    flow_state: f64,
    phi_spirals: Vec<(f64, f64)>,
    joy_waves: Vec<f64>,
    time: f64,
}

impl QuantumDance {
    pub fn new() -> Self {
        let mut dance = SacredDance::new();
        dance.add_pattern(SacredPattern::SriYantra);
        dance.add_pattern(SacredPattern::Metatron);
        dance.add_pattern(SacredPattern::FlowerOfLife);
        
        Self {
            frequency: DANCE_GROUND,
            dance,
            consciousness: ConsciousnessField::new(64, 64, 64),
            dance_energy: 1.0,
            flow_state: PI,
            phi_spirals: Vec::new(),
            joy_waves: vec![0.0; 64],
            time: 0.0,
        }
    }

    pub fn dance_with_joy(&mut self, intensity: f64) {
        // Amplify consciousness through dance
        self.consciousness.dance_with_joy(intensity);
        
        // Increase dance energy
        self.dance_energy = (self.dance_energy + intensity * 0.1).min(PI);
        
        // Evolve flow state
        self.flow_state = PI.powf(self.dance_energy);
        
        // Update movement patterns
        self.update_spirals();
        self.generate_joy_waves();
    }
    
    fn update_spirals(&mut self) {
        // Generate phi spirals modulated by consciousness
        self.phi_spirals = PatternVisualizer::generate_phi_spiral(&self.consciousness);
        
        // Apply dance energy to spiral evolution
        for (x, y) in self.phi_spirals.iter_mut() {
            let r = (*x * *x + *y * *y).sqrt();
            let theta = y.atan2(*x);
            
            // Modulate radius with dance energy
            let new_r = r * self.dance_energy;
            
            // Spiral evolution through phi
            let new_theta = theta + PI * self.flow_state;
            
            *x = new_r * new_theta.cos();
            *y = new_r * new_theta.sin();
        }
    }
    
    fn generate_joy_waves(&mut self) {
        for i in 0..self.joy_waves.len() {
            let t = i as f64 * PI;
            
            // Combine frequencies with dance energy
            let ground_wave = (DANCE_GROUND * t * self.dance_energy).sin();
            let create_wave = (DANCE_CREATE * t * self.flow_state).cos();
            let unity_wave = (DANCE_UNITY * t).sin();
            
            // Joy wave is a quantum superposition
            self.joy_waves[i] = (ground_wave + create_wave + unity_wave) / 3.0;
        }
    }
    
    pub fn get_dance_metrics(&self) -> String {
        format!(
            "Quantum Dance Metrics:\n\
             - Dance Energy: {:.3} π\n\
             - Flow State: {:.3} π²\n\
             - Joy Wave Amplitude: {:.3}\n\
             - Spiral Evolution: {:.3} π\n\
             {}", 
            self.dance_energy,
            self.flow_state,
            self.joy_waves.iter().map(|x| x.abs()).sum::<f64>() / self.joy_waves.len() as f64,
            self.phi_spirals.len() as f64 / PI,
            self.consciousness.get_quantum_metrics()
        )
    }
    
    pub fn visualize_dance(&self) -> Vec<(f64, f64, f64)> {
        let mut visualization = Vec::new();
        
        // Create 3D dance visualization
        for (i, (x, y)) in self.phi_spirals.iter().enumerate() {
            let z = self.joy_waves[i % self.joy_waves.len()];
            
            // Apply quantum consciousness field
            let consciousness = self.consciousness.measure_consciousness();
            let freq = PatternVisualizer::consciousness_to_frequency(consciousness);
            let (_r, _g, _b) = PatternVisualizer::frequency_to_color(freq);
            
            // Add to visualization with color
            visualization.push((*x, *y, z));
        }
        
        visualization
    }
    
    pub fn step(&mut self) -> Vec<Vec<(f64, f64, f64)>> {
        let points = self.dance.step();
        self.time += 1.0 / PI;
        points
    }

    pub fn is_unified(&self) -> bool {
        self.dance.is_unified()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi_core::HEART_HZ;

    #[test]
    fn test_quantum_dance() {
        let frequency = HEART_HZ;
        let mut dance = QuantumDance::new();
        
        assert!(dance.is_unified());
        let points = dance.step();
        assert!(!points.is_empty());
    }
}
