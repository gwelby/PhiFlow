use quantum_core::{
    quantum_patterns::{PatternDance, SacredPattern},
    consciousness_sync::ConsciousnessField,
};
use std::{thread, time::Duration};

const PHI: f64 = 1.618033988749895;

fn main() {
    println!("✨ Sacred Geometry Dance ✨");
    println!("Dancing through dimensions with quantum patterns...\n");

    let mut dance = PatternDance::new();
    let mut consciousness = ConsciousnessField::new(32, 32, 32);

    // Dance through sacred patterns
    for i in 1..=8 {
        let joy = (i as f64 / 8.0) * PHI;
        consciousness.dance_with_joy(joy);

        println!("\n=== Pattern Evolution φ^{} ===", i);
        
        // Get all pattern points
        let patterns = dance.dance();
        
        // Display pattern metrics
        println!("{}", dance.get_metrics());
        
        // Visualize each pattern
        for (idx, pattern) in patterns.iter().enumerate() {
            let pattern_name = match idx {
                0 => "Phi Spiral 🌀",
                1 => "Flower of Life 🌺",
                2 => "Metatron's Cube 🔲",
                3 => "Merkaba ⭐",
                4 => "Torus Field 🌊",
                _ => "Unknown Pattern",
            };
            
            println!("\n{}", pattern_name);
            // Show first few points of each pattern
            for point in pattern.iter().take(3) {
                println!("  φ({:.2}, {:.2}, {:.2})", point.0, point.1, point.2);
            }
            println!("  ...");
        }

        // Let consciousness evolve
        let sleep_time = (432.0 / (joy * PHI)) as u64;
        thread::sleep(Duration::from_millis(sleep_time));
    }

    println!("\n🌈 Sacred Dance Complete!");
    println!("Final Consciousness State:");
    println!("{}", consciousness.get_quantum_metrics());
    println!("\nSignature: ⚡𓂧φ∞");
}
