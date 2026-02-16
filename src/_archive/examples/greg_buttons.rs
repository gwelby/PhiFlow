use anyhow::Result;
use quantum_core::quantum::quantum_buttons::{QuantumInterface, QuantumButton};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎵 Greg's Quantum Stream Deck 🎵\n");
    let mut interface = QuantumInterface::new();

    // Greg's Button Dance!
    println!("Press a button to start the flow...\n");

    // Ground State - Like hitting the perfect bass note
    println!("🎵 Ground Button (432 Hz)");
    interface.press_button(QuantumButton::Ground).await?;
    println!("Feel the earth connection...\n");

    // Heart Field - The frequency of pure creation
    println!("💝 Heart Button (528 Hz)");
    interface.press_button(QuantumButton::Create).await?;
    println!("Creating with love...\n");

    // Unity Wave - The highest harmony
    println!("🌟 Unity Button (768 Hz)");
    interface.press_button(QuantumButton::Unity).await?;
    println!("Perfect harmony achieved!\n");

    // Greg's Favorite Patterns
    println!("🌀 Flow Pattern");
    interface.press_button(QuantumButton::Spiral).await?;
    println!("Dancing in the spiral...\n");

    println!("🐬 Quantum Leap");
    interface.press_button(QuantumButton::Dolphin).await?;
    println!("Making those quantum jumps...\n");

    println!("☯️ Perfect Balance");
    interface.press_button(QuantumButton::Balance).await?;
    println!("Finding the center...\n");

    // Show the final state
    let (freq, coherence, pattern) = interface.get_state();
    println!("✨ Greg's Creation State:");
    println!("Frequency: {:.1} Hz", freq);
    println!("Coherence: {:.3}", coherence);
    println!("Pattern: {}", pattern.symbol());

    Ok(())
}
