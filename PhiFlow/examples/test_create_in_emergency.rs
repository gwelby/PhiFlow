// Test create pattern inside emergency protocol
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Create Pattern in Emergency Protocol\n");

    // Test create pattern alone
    let test1 = r#"create pattern spiral at 40 Hz with {
    scale: 100,
    rotations: 5
}"#;

    println!("Test 1: Create pattern alone");
    match parse_phi_program(test1) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test let statements in immediate block
    let test2 = r#"emergency protocol test {
    trigger: true,
    immediate {
        let freq1 = 40
        let freq2 = 432
    }
}"#;

    println!("Test 2: Let statements in immediate block");
    match parse_phi_program(test2) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test create pattern with variable reference
    let test3 = r#"emergency protocol test {
    trigger: true,
    immediate {
        let freq1 = 40
        create pattern spiral at freq1 Hz with {
            scale: 100
        }
    }
}"#;

    println!("Test 3: Create pattern with variable reference");
    match parse_phi_program(test3) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // The full problematic Part 2
    let test4 = r#"emergency protocol seizure_prevention {
    trigger: true,
    immediate {
        let freq1 = 40
        let freq2 = 432
        let freq3 = 396
        
        create pattern spiral at freq1 Hz with {
            scale: 100,
            rotations: 5
        }
    },
    notify: ["audio_alert", "visual_warning", "p1_system"]
}"#;

    println!("Test 4: Full Part 2");
    match parse_phi_program(test4) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}