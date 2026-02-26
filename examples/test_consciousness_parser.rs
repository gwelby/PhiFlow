// Test Consciousness Parser
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing PhiFlow Consciousness Parser\n");

    // Test 1: Basic consciousness declaration
    let program1 = r#"
consciousness TRANSCEND {
    frequency: 720Hz,
    coherence: 0.867,
    intention: "quantum tunneling perception"
}"#;

    println!("Test 1: Consciousness Declaration");
    println!("Program:\n{}", program1);
    match parse_phi_program(program1) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("AST: {:#?}\n", ast);
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }

    // Test 2: Hardware declaration
    let program2 = r#"
hardware rgb_keyboard {
    device: "HCY-K016"
}"#;

    println!("Test 2: Hardware Declaration");
    println!("Program:\n{}", program2);
    match parse_phi_program(program2) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("AST: {:#?}\n", ast);
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }

    // Test 3: Emergency protocol
    let program3 = r#"
emergency protocol seizure_prevention {
    trigger: true,
    immediate {
        let x = 40
    },
    notify: ["audio_alert", "visual_warning"]
}"#;

    println!("Test 3: Emergency Protocol");
    println!("Program:\n{}", program3);
    match parse_phi_program(program3) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("AST: {:#?}\n", ast);
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }

    // Test 4: Multiple consciousness constructs
    let program4 = r#"
consciousness OBSERVE {
    frequency: 432Hz,
    coherence: 1.0,
    intention: "ground state awareness"
}

consciousness CREATE {
    frequency: 528Hz,
    coherence: 0.9,
    intention: "manifest reality"
}

biological_program heal_dna {
    target: "human_dna",
    frequency: 528Hz
}"#;

    println!("Test 4: Multiple Consciousness Constructs");
    println!("Program:\n{}", program4);
    match parse_phi_program(program4) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("Number of expressions: {}", ast.len());
            for (i, expr) in ast.iter().enumerate() {
                println!("Expression {}: {:#?}", i + 1, expr);
            }
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }
}