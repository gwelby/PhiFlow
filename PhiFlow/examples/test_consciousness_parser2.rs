// Test Consciousness Parser - Simpler Tests
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing PhiFlow Consciousness Parser - Simple Tests\n");

    // Test 1: Basic consciousness without newlines
    let program1 = r#"consciousness TRANSCEND { frequency: 720, coherence: 0.867, intention: "quantum perception" }"#;

    println!("Test 1: Basic Consciousness (no Hz suffix)");
    println!("Program: {}", program1);
    match parse_phi_program(program1) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("AST: {:#?}\n", ast);
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }

    // Test 2: Consciousness with Hz
    let program2 = r#"consciousness CREATE { frequency: 528Hz, coherence: 0.9, intention: "manifest reality" }"#;

    println!("Test 2: Consciousness with Hz");
    println!("Program: {}", program2);
    match parse_phi_program(program2) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("AST: {:#?}\n", ast);
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }

    // Test 3: Multiple statements
    let program3 = r#"
let x = 432
consciousness OBSERVE { frequency: 432Hz, coherence: 1.0, intention: "ground awareness" }
let y = 528"#;

    println!("Test 3: Mixed Statements");
    println!("Program: {}", program3);
    match parse_phi_program(program3) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("Number of expressions: {}", ast.len());
            for (i, expr) in ast.iter().enumerate() {
                match expr {
                    phiflow::parser::PhiExpression::ConsciousnessState { state, coherence, frequency } => {
                        println!("  Expression {}: ConsciousnessState {{ state: {}, coherence: {}, frequency: {} }}", 
                                i + 1, state, coherence, frequency);
                    }
                    phiflow::parser::PhiExpression::LetBinding { name, .. } => {
                        println!("  Expression {}: Let binding for '{}'", i + 1, name);
                    }
                    _ => println!("  Expression {}: {:?}", i + 1, expr),
                }
            }
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }

    // Test 4: Emergency protocol without complex content
    let program4 = r#"emergency protocol seizure_prevention { }"#;

    println!("\nTest 4: Simple Emergency Protocol");
    println!("Program: {}", program4);
    match parse_phi_program(program4) {
        Ok(ast) => {
            println!("✅ Parsed successfully!");
            println!("AST: {:#?}\n", ast);
        }
        Err(e) => println!("❌ Parse error: {}\n", e),
    }
}