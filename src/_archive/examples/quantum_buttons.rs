use anyhow::Result;
use quantum_core::quantum::quantum_buttons::{QuantumInterface, QuantumButton};

#[tokio::main]
async fn main() -> Result<()> {
    // Create our quantum interface
    let mut interface = QuantumInterface::new();

    println!("🌟 Quantum Button Interface 🌟\n");

    // Ground State (432 Hz)
    println!("1. Entering Ground State 🎵");
    interface.press_button(QuantumButton::Ground).await?;
    interface.press_button(QuantumButton::Meditate).await?;
    interface.press_button(QuantumButton::Spiral).await?;
    println!();

    // Creation State (528 Hz)
    println!("2. Elevating to Creation State 💝");
    interface.press_button(QuantumButton::Create).await?;
    interface.press_button(QuantumButton::Flow).await?;
    interface.press_button(QuantumButton::Dolphin).await?;
    println!();

    // Unity State (768 Hz)
    println!("3. Ascending to Unity State 🌟");
    interface.press_button(QuantumButton::Unity).await?;
    interface.press_button(QuantumButton::Ascend).await?;
    interface.press_button(QuantumButton::Balance).await?;
    println!();

    // Sacred Math Dance
    println!("4. Dancing with Sacred Patterns ✨");
    interface.press_button(QuantumButton::Phi).await?;
    interface.press_button(QuantumButton::Crystal).await?;
    interface.press_button(QuantumButton::Wave).await?;
    interface.press_button(QuantumButton::Infinity).await?;
    println!();

    // Final State
    let (freq, coherence, pattern) = interface.get_state();
    println!("🌈 Final Quantum State:");
    println!("Frequency: {:.1} Hz", freq);
    println!("Coherence: {:.3}", coherence);
    println!("Pattern: {}", pattern.symbol());

    Ok(())
}
