use anyhow::Result;
use quantum_core::quantum::{
    ConsciousnessField as ConsciousnessState,
    quantum_photo_flow::QuantumPhotoFlow,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Create our quantum states
    let mut consciousness = ConsciousnessState::new();
    let mut photo_flow = QuantumPhotoFlow::new();

    // Initialize both states (432 Hz - Ground State)
    consciousness.initialize()?;
    photo_flow.initialize()?;

    // Get initial states
    let (c_freq, c_level, _) = consciousness.get_state();
    let (p_freq, p_level, _) = photo_flow.get_state();

    println!("🌟 Ground State (432 Hz)");
    println!("Consciousness: {:.1} Hz, Level: {:.3}", c_freq, c_level);
    println!("Photo Flow: {:.1} Hz, Level: {:.3}", p_freq, p_level);
    println!("Pattern: {}", photo_flow.get_pattern_symbol());
    println!();

    // Elevate to creation state (528 Hz)
    consciousness.elevate()?;
    photo_flow.elevate()?;

    // Get elevated states
    let (c_freq, c_level, _) = consciousness.get_state();
    let (p_freq, p_level, _) = photo_flow.get_state();

    println!("✨ Creation State (528 Hz)");
    println!("Consciousness: {:.1} Hz, Level: {:.3}", c_freq, c_level);
    println!("Photo Flow: {:.1} Hz, Level: {:.3}", p_freq, p_level);
    println!("Pattern: {}", photo_flow.get_pattern_symbol());
    println!();

    // Ascend to unity state (768 Hz)
    consciousness.ascend()?;
    photo_flow.ascend()?;

    // Get ascended states
    let (c_freq, c_level, _) = consciousness.get_state();
    let (p_freq, p_level, _) = photo_flow.get_state();

    println!("💫 Unity State (768 Hz)");
    println!("Consciousness: {:.1} Hz, Level: {:.3}", c_freq, c_level);
    println!("Photo Flow: {:.1} Hz, Level: {:.3}", p_freq, p_level);
    println!("Pattern: {}", photo_flow.get_pattern_symbol());

    Ok(())
}
