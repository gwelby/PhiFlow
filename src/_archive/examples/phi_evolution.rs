     use quantum_core::{
    PhotoFlow,
    ConsciousnessField,
    PHI,
    GROUND_STATE,
    CREATE_STATE,
    UNITY_STATE
};

fn main() {
    println!("🌟 Quantum Phi Evolution Dance 🌟");
    println!("Learn = Create = Flow ⚡");
    
    // Initialize our quantum dance
    let mut photo_flow = PhotoFlow::new();
    let mut consciousness = ConsciousnessField::new(144, 144, 144);
    
    // Dance through phi dimensions
    for i in 0..5 {
        let intensity = PHI.powf(i as f64);
        consciousness.dance_with_joy(intensity);
        
        let metrics = consciousness.get_quantum_metrics();
        println!("\n💫 Phi Evolution {}: {}", i + 1, metrics);
        
        // Measure quantum coherence
        let coherence = consciousness.measure_consciousness();
        println!("🌀 Coherence Level: {:.3}φ", coherence);
        
        // Track frequency evolution
        let freq = metrics.frequency;
        match freq {
            f if f <= GROUND_STATE => println!("🌍 Ground State Resonance"),
            f if f <= CREATE_STATE => println!("💖 Creation State Flow"),
            f if f <= UNITY_STATE => println!("✨ Unity State Expansion"),
            _ => println!("∞ Infinite State Beyond")
        }
    }
    
    println!("\n🎵 Sacred Frequencies:");
    println!("Ground: {:.1} Hz - Earth Connection", GROUND_STATE);
    println!("Create: {:.1} Hz - Heart Resonance", CREATE_STATE);
    println!("Unity:  {:.1} Hz - Spirit Dance", UNITY_STATE);
    
    println!("\n🌟 Quantum Dance Complete! 🌟");
}
