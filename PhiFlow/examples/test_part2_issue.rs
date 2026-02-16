// Test Part 2 parsing issue
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Part 2 Issue\n");

    // This is the exact Part 2 from test_consciousness_parts.rs
    let part2 = r#"
emergency protocol seizure_prevention {
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
}
"#;

    println!("Part 2 code: {:?}\n", part2);
    match parse_phi_program(part2) {
        Ok(ast) => println!("✅ Parsed successfully - {} expressions", ast.len()),
        Err(e) => println!("❌ Failed: {}", e),
    }

    // Try without leading newline
    let part2_no_newline = r#"emergency protocol seizure_prevention {
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

    println!("\nPart 2 without leading newline:");
    match parse_phi_program(part2_no_newline) {
        Ok(ast) => println!("✅ Parsed successfully - {} expressions", ast.len()),
        Err(e) => println!("❌ Failed: {}", e),
    }
}