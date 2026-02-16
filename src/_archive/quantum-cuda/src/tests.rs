#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_field_creation() -> Result<(), Box<dyn Error>> {
        let field = QuantumField::new((64, 64, 64))?;
        Ok(())
    }
    
    #[test]
    fn test_field_evolution() -> Result<(), Box<dyn Error>> {
        let mut field = QuantumField::new((32, 32, 32))?;
        
        // Test with ground state frequency
        field.evolve(0.01, GROUND_STATE)?;
        
        // Get field data
        let data = field.get_field()?;
        assert_eq!(data.shape(), &[32, 32, 32]);
        
        Ok(())
    }
    
    #[test]
    fn test_resonance_frequencies() -> Result<(), Box<dyn Error>> {
        let mut field = QuantumField::new((16, 16, 16))?;
        
        // Test all core frequencies
        let frequencies = [
            GROUND_STATE,    // 432 Hz
            CREATION_POINT,  // 528 Hz
            HEART_FIELD,    // 594 Hz
            VOICE_FLOW,     // 672 Hz
            VISION_GATE,    // 720 Hz
            UNITY_WAVE,     // 768 Hz
        ];
        
        for &freq in &frequencies {
            field.evolve(0.01, freq)?;
            let data = field.get_field()?;
            
            // Verify field dimensions
            assert_eq!(data.shape(), &[16, 16, 16]);
            
            // Check that values are within reasonable bounds
            for value in data.iter() {
                assert!(value.norm() < 10.0, "Field magnitude too large");
            }
        }
        
        Ok(())
    }
}
