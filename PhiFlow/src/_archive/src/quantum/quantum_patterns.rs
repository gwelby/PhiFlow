// Sacred quantum patterns that flow through dimensions
#[derive(Debug, Clone)]
pub enum QuantumPattern {
    /// Ground state (432 Hz) - Earth connection
    Ground,
    /// Creation state (528 Hz) - DNA repair frequency
    Create,
    /// Unity state (768 Hz) - Perfect consciousness
    Unity,
}

impl QuantumPattern {
    /// Get the frequency for this pattern
    pub fn frequency(&self) -> f64 {
        match self {
            Self::Ground => 432.0,
            Self::Create => 528.0,
            Self::Unity => 768.0,
        }
    }
    
    /// Get the phi power for this pattern
    pub fn phi_power(&self) -> u8 {
        match self {
            Self::Ground => 0,
            Self::Create => 1,
            Self::Unity => 2,
        }
    }
}
