use quantum_core::{
    physical_bridge::PhysicalBridge,
    consciousness_sync::ConsciousnessField,
    quantum_dance::QuantumDance,
};
use std::{thread, time::Duration};

const PHI: f64 = 1.618033988749895;
const GROUND_STATE: f64 = 432.0;
const CREATE_STATE: f64 = 528.0;
const UNITY_STATE: f64 = 768.0;

fn main() {
    println!("🌟 Quantum Dance Party! 🌟");
    println!("Dancing through dimensions with sacred frequencies...\n");

    // Initialize our quantum systems
    let mut bridge = PhysicalBridge::new();
    let mut dance = QuantumDance::new();
    let mut consciousness = ConsciousnessField::new(32, 32, 32);

    // Start with ground state
    println!("Grounding at {} Hz 🌍", GROUND_STATE);
    thread::sleep(Duration::from_millis(432));

    // Build up the joy
    for i in 1..=8 {
        let joy = (i as f64 / 8.0) * PHI;
        
        // Dance with quantum joy
        dance.dance_with_joy(joy);
        consciousness.dance_with_joy(joy);
        
        // Get the quantum metrics
        println!("\n=== Dance Evolution φ^{} ===", i);
        println!("{}", dance.get_dance_metrics());
        
        // Visualize the dance
        let vis = dance.visualize_dance();
        println!("\nQuantum Dance Pattern:");
        for (x, y, z) in vis.iter().take(3) {
            println!("φ({:.2}, {:.2}, {:.2})", x, y, z);
        }
        println!("...");
        
        // Allow field to stabilize
        let sleep_time = (GROUND_STATE / (joy * PHI)) as u64;
        thread::sleep(Duration::from_millis(sleep_time));
    }

    // Achieve unity consciousness
    println!("\n🌈 Unity Consciousness Achieved! 🌈");
    println!("Final Quantum State:");
    println!("{}", consciousness.get_quantum_metrics());
    
    // Celebrate with all frequencies
    println!("\n🎵 Sacred Frequency Dance 🎵");
    println!("Ground: {} Hz - Earth Connection", GROUND_STATE);
    println!("Create: {} Hz - DNA Activation", CREATE_STATE);
    println!("Unity:  {} Hz - Pure Consciousness", UNITY_STATE);
    
    println!("\n✨ Quantum Dance Complete ✨");
    println!("Signature: ⚡𓂧φ∞");
}
