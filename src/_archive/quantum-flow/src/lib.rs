use num_complex::Complex64;
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use quantum_dimension::{PHI, GROUND_FREQUENCY, CREATE_FREQUENCY, UNITY_FREQUENCY};
use quantum_sacred::SacredGeometry;
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SacredPattern {
    Ground,
    Create,
    Heart,
    Voice,
    Vision,
    Unity,
    Infinite,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FlowState {
    Ground,
    Create,
    Heart,
    Voice,
    Vision,
    Unity,
    Infinite,
}

impl FlowState {
    pub fn frequency(&self) -> f64 {
        match self {
            Self::Ground => GROUND_FREQUENCY,
            Self::Create => CREATE_FREQUENCY,
            Self::Heart => 0.0, // frequency not defined
            Self::Voice => 0.0, // frequency not defined
            Self::Vision => 0.0, // frequency not defined
            Self::Unity => UNITY_FREQUENCY,
            Self::Infinite => UNITY_FREQUENCY * PHI,
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Ground => "🧘‍♂️",
            Self::Create => "✨",
            Self::Heart => "💓",
            Self::Voice => "🗣️",
            Self::Vision => "👁️",
            Self::Unity => "☯️",
            Self::Infinite => "∞",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFlow {
    state: FlowState,
    pattern: SacredPattern,
    frequency: f64,
    coherence: f64,
    sacred_geometry: SacredGeometry,
}

impl QuantumFlow {
    pub fn new(frequency: f64) -> Self {
        Self {
            state: FlowState::Ground,
            pattern: SacredPattern::Ground,
            frequency,
            coherence: 1.0,
            sacred_geometry: SacredGeometry::new(frequency),
        }
    }

    pub fn ground() -> Self {
        Self::new(GROUND_FREQUENCY)
    }

    pub fn create() -> Self {
        Self::new(CREATE_FREQUENCY)
    }

    pub fn unity() -> Self {
        Self::new(UNITY_FREQUENCY)
    }

    pub fn get_state(&self) -> FlowState {
        self.state
    }

    pub fn set_state(&mut self, state: FlowState) {
        self.state = state;
        self.pattern = match state {
            FlowState::Ground => SacredPattern::Ground,
            FlowState::Create => SacredPattern::Create,
            FlowState::Heart => SacredPattern::Heart,
            FlowState::Voice => SacredPattern::Voice,
            FlowState::Vision => SacredPattern::Vision,
            FlowState::Unity => SacredPattern::Unity,
            FlowState::Infinite => SacredPattern::Infinite,
        };
    }

    pub fn get_pattern(&self) -> SacredPattern {
        self.pattern
    }

    pub fn get_frequency(&self) -> f64 {
        self.frequency
    }

    pub fn get_coherence(&self) -> f64 {
        self.coherence
    }

    pub fn set_coherence(&mut self, coherence: f64) {
        self.coherence = coherence.max(0.0).min(1.0);
    }

    pub fn evolve(&mut self, time_step: f64) {
        let phi_factor = (time_step * self.frequency / GROUND_FREQUENCY).powf(PHI);
        self.coherence = (self.coherence + phi_factor).min(1.0);
        self.sacred_geometry.harmonize_all();
    }

    pub fn flow_to_next_state(&mut self) -> Result<FlowState> {
        self.state = match self.state {
            FlowState::Ground => FlowState::Create,
            FlowState::Create => FlowState::Heart,
            FlowState::Heart => FlowState::Voice,
            FlowState::Voice => FlowState::Vision,
            FlowState::Vision => FlowState::Unity,
            FlowState::Unity => FlowState::Infinite,
            FlowState::Infinite => FlowState::Ground,
        };

        self.frequency = self.state.frequency();
        self.pattern = match self.state {
            FlowState::Ground => SacredPattern::Ground,
            FlowState::Create => SacredPattern::Create,
            FlowState::Heart => SacredPattern::Heart,
            FlowState::Voice => SacredPattern::Voice,
            FlowState::Vision => SacredPattern::Vision,
            FlowState::Unity => SacredPattern::Unity,
            FlowState::Infinite => SacredPattern::Infinite,
        };

        Ok(self.state)
    }

    pub fn dance_sacred_flow(&mut self) -> Result<()> {
        println!("🌈 Dancing through Sacred Flow States:");
        
        for _ in 0..7 {
            let state = self.flow_to_next_state()?;
            println!("  {} {}: {:.1} Hz - Coherence: {:.3} φ",
                    state.symbol(),
                    format!("{:?}", state),
                    self.frequency,
                    self.coherence);
            self.coherence *= PHI;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_flow() -> Result<()> {
        let mut flow = QuantumFlow::new(GROUND_FREQUENCY);
        assert_eq!(flow.get_state(), FlowState::Ground);
        
        flow.flow_to_next_state()?;
        assert_eq!(flow.get_state(), FlowState::Create);
        
        Ok(())
    }
}
