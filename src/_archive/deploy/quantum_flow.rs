use serde::{Serialize, Deserialize};
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FlowState {
    Ground,  // 432 Hz
    Create,  // 528 Hz
    Heart,   // 594 Hz
    Voice,   // 672 Hz
    Vision,  // 720 Hz
    Unity,   // 768 Hz
    Infinite // φ^φ
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFlow {
    frequency: f64,
    state: FlowState,
    coherence: f64,
}

impl QuantumFlow {
    pub fn new() -> Self {
        Self {
            frequency: 432.0, // Start at ground state
            state: FlowState::Ground,
            coherence: 1.0,
        }
    }

    pub fn create_sequence(&mut self, target: FlowState) -> Result<()> {
        self.state = target;
        self.frequency = match target {
            FlowState::Ground => 432.0,
            FlowState::Create => 528.0,
            FlowState::Heart => 594.0,
            FlowState::Voice => 672.0,
            FlowState::Vision => 720.0,
            FlowState::Unity => 768.0,
            FlowState::Infinite => 1440.0, // φ^φ * 432
        };
        Ok(())
    }

    pub fn evolve_all(&mut self) -> bool {
        // Evolve through states based on phi ratio
        const PHI: f64 = 1.618033988749895;
        self.frequency *= PHI;
        
        // Update state based on new frequency
        self.state = if self.frequency >= 1440.0 {
            FlowState::Infinite
        } else if self.frequency >= 768.0 {
            FlowState::Unity
        } else if self.frequency >= 720.0 {
            FlowState::Vision
        } else if self.frequency >= 672.0 {
            FlowState::Voice
        } else if self.frequency >= 594.0 {
            FlowState::Heart
        } else if self.frequency >= 528.0 {
            FlowState::Create
        } else {
            FlowState::Ground
        };

        true
    }

    pub fn get_metrics(&self) -> String {
        format!(
            "QUANTUM FLOW METRICS\n\
            ==================\n\
            Frequency: {:.2} Hz\n\
            State: {:?}\n\
            Coherence: {:.6}",
            self.frequency,
            self.state,
            self.coherence
        )
    }
}
