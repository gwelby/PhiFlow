// Test simple emergency protocol
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Simple Emergency Protocol\n");

    // Very simple emergency protocol
    let test1 = r#"emergency protocol test {
    trigger: true
}"#;

    println!("Test 1: Simple trigger only");
    match parse_phi_program(test1) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // With immediate block
    let test2 = r#"emergency protocol test {
    trigger: true,
    immediate { }
}"#;

    println!("Test 2: With empty immediate block");
    match parse_phi_program(test2) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // With notify
    let test3 = r#"emergency protocol test {
    trigger: true,
    immediate { },
    notify: []
}"#;

    println!("Test 3: With notify");
    match parse_phi_program(test3) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // With notify but no trailing comma
    let test4 = r#"emergency protocol test {
    notify: []
}"#;

    println!("Test 4: Just notify, no comma");
    match parse_phi_program(test4) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Debug - what happens with trailing comma
    let test5 = r#"emergency protocol test {
    trigger: true,
}"#;

    println!("Test 5: Trailing comma after trigger");
    match parse_phi_program(test5) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}