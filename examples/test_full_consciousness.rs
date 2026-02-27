// Test parsing a full consciousness program
use phiflow::parser::parse_phi_program;
use std::fs;

fn main() {
    println!("🧪 Testing Full Consciousness Program Parsing\n");

    // Read the consciousness program
    let program = fs::read_to_string("examples/consciousness_seizure_prevention.phi")
        .expect("Failed to read consciousness program");
    
    println!("Program:");
    println!("{}", "=".repeat(80));
    println!("{}", program);
    println!("{}", "=".repeat(80));
    
    match parse_phi_program(&program) {
        Ok(ast) => {
            println!("\n✅ Parsed successfully!");
            println!("\nAST contains {} top-level expressions", ast.len());
            
            // Count different types of expressions
            let mut consciousness_count = 0;
            let mut hardware_count = 0;
            let mut emergency_count = 0;
            let mut biological_count = 0;
            let mut function_count = 0;
            let mut other_count = 0;
            
            for expr in &ast {
                match expr {
                    phiflow::parser::PhiExpression::ConsciousnessState { .. } => consciousness_count += 1,
                    phiflow::parser::PhiExpression::HardwareSync { .. } => hardware_count += 1,
                    phiflow::parser::PhiExpression::EmergencyProtocol { .. } => emergency_count += 1,
                    phiflow::parser::PhiExpression::BiologicalInterface { .. } => biological_count += 1,
                    phiflow::parser::PhiExpression::FunctionDef { .. } => function_count += 1,
                    _ => other_count += 1,
                }
            }
            
            println!("\nExpression types:");
            println!("  Consciousness states: {}", consciousness_count);
            println!("  Hardware declarations: {}", hardware_count);
            println!("  Emergency protocols: {}", emergency_count);
            println!("  Biological programs: {}", biological_count);
            println!("  Function definitions: {}", function_count);
            println!("  Other expressions: {}", other_count);
        }
        Err(e) => {
            println!("\n❌ Parse error: {}", e);
        }
    }
}