use anyhow::Result;
use serde::{Serialize, Deserialize};
use super::quantum_constants::{
    GROUND_FREQUENCY,
    CREATE_FREQUENCY,
    HEART_FREQUENCY,
    VOICE_FREQUENCY,
    VISION_FREQUENCY,
    UNITY_FREQUENCY,
    PHI,
    PHI_SQUARED,
    PHI_CUBED,
    HUMAN_SCALE,
    SacredPattern,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPhotoFlow {
    frequency: f64,
    coherence: f64,
    resonance: f64,
    pattern: SacredPattern,
    flow_state: FlowState,
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
            Self::Heart => HEART_FREQUENCY,
            Self::Voice => VOICE_FREQUENCY,
            Self::Vision => VISION_FREQUENCY,
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

impl QuantumPhotoFlow {
    pub fn new(frequency: f64) -> Result<Self> {
        let mut flow = Self {
            frequency,
            coherence: HUMAN_SCALE,
            resonance: PHI,
            pattern: SacredPattern::Ground,
            flow_state: FlowState::Ground,
        };
        flow.initialize()?;
        Ok(flow)
    }

    fn initialize(&mut self) -> Result<()> {
        self.resonance = match self.frequency {
            f if f == GROUND_FREQUENCY => PHI,
            f if f == CREATE_FREQUENCY => PHI_SQUARED,
            f if f == UNITY_FREQUENCY => PHI_CUBED,
            _ => PHI,
        };
        Ok(())
    }

    pub fn flow_to_next_state(&mut self) -> Result<FlowState> {
        self.flow_state = match self.flow_state {
            FlowState::Ground => FlowState::Create,
            FlowState::Create => FlowState::Heart,
            FlowState::Heart => FlowState::Voice,
            FlowState::Voice => FlowState::Vision,
            FlowState::Vision => FlowState::Unity,
            FlowState::Unity => FlowState::Infinite,
            FlowState::Infinite => FlowState::Ground,
        };

        self.frequency = self.flow_state.frequency();
        self.pattern = match self.flow_state {
            FlowState::Ground => SacredPattern::Ground,
            FlowState::Create => SacredPattern::Create,
            FlowState::Heart => SacredPattern::Heart,
            FlowState::Voice => SacredPattern::Voice,
            FlowState::Vision => SacredPattern::Vision,
            FlowState::Unity => SacredPattern::Unity,
            FlowState::Infinite => SacredPattern::Infinite,
        };

        println!("🌊 Flowing to: {} at {:.1} Hz", 
                self.flow_state.symbol(), 
                self.frequency);

        Ok(self.flow_state)
    }

    pub fn get_flow_state(&self) -> FlowState {
        self.flow_state
    }

    pub fn get_pattern(&self) -> SacredPattern {
        self.pattern
    }

    pub fn dance_sacred_flow(&mut self) -> Result<()> {
        println!("🌈 Dancing through Sacred Flow States:");
        
        for _ in 0..7 {
            let state = self.flow_to_next_state()?;
            println!("  {} {}: {:.1} Hz - Coherence: {:.3} φ",
                    state.symbol(),
                    format!("{:?}", state),
                    self.frequency,
                    self.resonance);
            self.resonance *= PHI;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_flow() -> Result<()> {
        let mut flow = QuantumPhotoFlow::new(GROUND_FREQUENCY)?;
        assert_eq!(flow.get_flow_state(), FlowState::Ground);
        
        flow.flow_to_next_state()?;
        assert_eq!(flow.get_flow_state(), FlowState::Create);
        
        Ok(())
    }
}
