// Test function body parsing
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Function Body Parsing\n");

    // Test 1: Function with inline body
    let test1 = r#"function test() -> Number { return 1.0 }"#;
    
    println!("Test 1: Inline body");
    match parse_phi_program(test1) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 2: Function with body on next line (no space)
    let test2 = r#"function test() -> Number
{ return 1.0 }"#;
    
    println!("Test 2: Body on next line (no space)");
    match parse_phi_program(test2) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 3: Function with body on next line (with space/newline)
    let test3 = r#"function test() -> Number 
{
    return 1.0
}"#;
    
    println!("Test 3: Body on next line with newline");
    match parse_phi_program(test3) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test 4: Function without return type
    let test4 = r#"function test() {
    return 1.0
}"#;
    
    println!("Test 4: No return type");
    match parse_phi_program(test4) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}