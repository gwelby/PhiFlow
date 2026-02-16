// Test minimal consciousness program
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Minimal Consciousness Program\n");

    // Test each construct one by one
    let programs = vec![
        ("Consciousness", r#"consciousness OBSERVE { frequency: 432Hz, coherence: 1.0, intention: "test" }"#),
        ("Hardware", r#"hardware rgb_keyboard { device: "HCY-K016" }"#),
        ("Emergency", r#"emergency protocol test { trigger: true, immediate { }, notify: [] }"#),
        ("ConsciousnessFlow", r#"consciousness_flow { gradient consciousness.level { } }"#),
        ("BiologicalProgram", r#"biological_program heal { target: "dna", frequency: 528Hz, sequence: "ATCG" }"#),
        ("Function", r#"function test() -> Number { return 1.0 }"#),
        ("Variable", r#"let x = 1.0"#),
        ("If", r#"if true { let y = 1 }"#),
        ("Create", r#"create pattern spiral at 432 Hz with { scale: 100 }"#),
    ];

    for (name, program) in programs {
        println!("Testing {}: {}", name, program);
        match parse_phi_program(program) {
            Ok(_) => println!("✅ {} parsed successfully\n", name),
            Err(e) => println!("❌ {} failed: {}\n", name, e),
        }
    }
}