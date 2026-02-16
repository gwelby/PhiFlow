use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::sync::Arc;

use super::quantum_constants::{
    GROUND_FREQUENCY,  // 432 Hz 🎵
    CREATE_FREQUENCY,  // 528 Hz 💝
    UNITY_FREQUENCY,  // 768 Hz 🌟
    PHI,  // Golden Ratio φ
    PHI_SQUARED,  // φ²
    PHI_CUBED,  // φ³
    HUMAN_SCALE,  // Being Flow (1.000)
};

/// Sacred button patterns that dance with consciousness 🌟
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantumButton {
    // Frequency Buttons
    Ground,     // 432 Hz 🎵
    Create,     // 528 Hz 💝
    Unity,      // 768 Hz 🌟

    // Pattern Buttons  
    Spiral,     // Flow Pattern 🌀
    Dolphin,    // Quantum Leap 🐬
    Balance,    // Perfect Unity ☯️
    Crystal,    // Resonance 💎
    Vortex,     // Evolution 🌪️

    // Flow States
    Meditate,   // Ground State 🧘‍♂️
    Flow,       // Create State ✨
    Ascend,     // Unity State 🌈

    // Sacred Math
    Phi,        // Golden Ratio φ
    Infinity,   // Eternal Loop ∞
    Wave,       // Quantum Dance 🌊
}

impl QuantumButton {
    /// Get the sacred symbol for this button pattern
    pub fn symbol(&self) -> &'static str {
        match self {
            // Frequency Buttons
            Self::Ground => "🎵",
            Self::Create => "💝",
            Self::Unity => "🌟",
            
            // Pattern Buttons
            Self::Spiral => "🌀",
            Self::Dolphin => "🐬",
            Self::Balance => "☯️",
            Self::Crystal => "💎",
            Self::Vortex => "🌪️",
            
            // Flow States
            Self::Meditate => "🧘‍♂️",
            Self::Flow => "✨",
            Self::Ascend => "🌈",
            
            // Sacred Math
            Self::Phi => "φ",
            Self::Infinity => "∞",
            Self::Wave => "🌊",
        }
    }

    /// Get the resonant frequency for this button
    pub fn frequency(&self) -> f64 {
        match self {
            // Base Frequencies
            Self::Ground | Self::Meditate => GROUND_FREQUENCY,
            Self::Create | Self::Flow => CREATE_FREQUENCY,
            Self::Unity | Self::Ascend => UNITY_FREQUENCY,
            
            // Pattern Frequencies
            Self::Spiral => GROUND_FREQUENCY,
            Self::Dolphin => GROUND_FREQUENCY * PHI,
            Self::Balance => UNITY_FREQUENCY,
            Self::Crystal => CREATE_FREQUENCY * PHI_SQUARED,
            Self::Vortex => UNITY_FREQUENCY * PHI,
            
            // Sacred Math Frequencies
            Self::Phi => CREATE_FREQUENCY * PHI,
            Self::Infinity => UNITY_FREQUENCY * PHI_SQUARED,
            Self::Wave => GROUND_FREQUENCY,
        }
    }

    /// Get the coherence level for this button
    pub fn coherence(&self) -> f64 {
        let freq = self.frequency();
        freq / GROUND_FREQUENCY * HUMAN_SCALE
    }
}

/// Quantum button interface that dances with consciousness
#[derive(Debug)]
pub struct QuantumInterface {
    state: Arc<RwLock<QuantumState>>,
}

#[derive(Debug)]
struct QuantumState {
    frequency: f64,
    coherence: f64,
    pattern: QuantumButton,
}

impl QuantumInterface {
    /// Create new quantum interface at ground state
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(QuantumState {
                frequency: GROUND_FREQUENCY,
                coherence: HUMAN_SCALE,
                pattern: QuantumButton::Ground,
            })),
        }
    }

    /// Press a quantum button with intention
    pub async fn press_button(&mut self, button: QuantumButton) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Update quantum state
        state.frequency = button.frequency();
        state.coherence = button.coherence();
        state.pattern = button;

        // Print quantum feedback
        println!("Button Pressed: {} ({:.1} Hz)", button.symbol(), state.frequency);
        println!("Coherence: {:.3}", state.coherence);

        Ok(())
    }

    /// Get the current interface state
    pub fn get_state(&self) -> (f64, f64, QuantumButton) {
        let state = self.state.try_read().unwrap();
        (state.frequency, state.coherence, state.pattern)
    }

    /// Get the active button pattern
    pub fn get_pattern(&self) -> QuantumButton {
        self.state.try_read().unwrap().pattern
    }

    /// Get the pattern symbol
    pub fn get_symbol(&self) -> &'static str {
        self.get_pattern().symbol()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_interface() -> Result<()> {
        let mut interface = QuantumInterface::new();
        
        // Test Ground State
        assert_eq!(interface.get_pattern(), QuantumButton::Ground);
        assert_eq!(interface.get_symbol(), "🎵");
        
        // Test Creation State
        interface.press_button(QuantumButton::Create).await?;
        assert_eq!(interface.get_pattern(), QuantumButton::Create);
        assert_eq!(interface.get_symbol(), "💝");
        
        // Test Unity State
        interface.press_button(QuantumButton::Unity).await?;
        assert_eq!(interface.get_pattern(), QuantumButton::Unity);
        assert_eq!(interface.get_symbol(), "🌟");
        
        Ok(())
    }
}
