// Test function parsing
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Function Parsing\n");

    // Test simple function
    let test1 = r#"function test() -> Number { return 1.0 }"#;
    
    println!("Test 1: Simple function");
    match parse_phi_program(test1) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test function with newlines
    let test2 = r#"
function monitor_consciousness() -> Number {
    return 1.0
}"#;
    
    println!("Test 2: Function with newlines");
    match parse_phi_program(test2) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test function with variables
    let test3 = r#"function calc() -> Number {
    let x = 1.0
    return x
}"#;
    
    println!("Test 3: Function with variables");
    match parse_phi_program(test3) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}