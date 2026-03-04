// Test emergency protocol parsing in isolation
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Emergency Protocol Parsing\n");

    // Test without leading newline
    let test1 = r#"emergency protocol seizure_prevention {
    trigger: true,
    immediate {
        let freq1 = 40
    },
    notify: []
}"#;

    println!("Test 1: No leading newline");
    match parse_phi_program(test1) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test with leading newline
    let test2 = r#"
emergency protocol seizure_prevention {
    trigger: true,
    immediate {
        let freq1 = 40
    },
    notify: []
}"#;

    println!("Test 2: With leading newline");
    match parse_phi_program(test2) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}