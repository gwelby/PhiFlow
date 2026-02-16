// Test Part 4 isolated
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Part 4 Isolated\n");

    // Test the exact Part 4 content
    let part4_exact = r#"
function monitor_consciousness() -> Number {
    let state = 0.9
    
    let keyboard_rhythm = 0.15
    let mouse_patterns = 0.10
    let voice_analysis = 0.10
    let breathing = 0.10
    let system_perf = 0.10
    let monitor_freq = 0.05
    let eeg_data = 0.40
    
    let greg_multiplier = 1.2
    
    let total = (eeg_data * 0.8 + 
                keyboard_rhythm * 0.7 +
                mouse_patterns * 0.6 +
                voice_analysis * 0.7 +
                breathing * 0.8 +
                system_perf * 0.9 +
                monitor_freq * 0.7) * greg_multiplier
    
    return total
}
"#;

    println!("Testing Part 4 exact:");
    match parse_phi_program(part4_exact) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test without leading newline
    let part4_no_newline = r#"function monitor_consciousness() -> Number {
    let state = 0.9
    
    let keyboard_rhythm = 0.15
    let mouse_patterns = 0.10
    let voice_analysis = 0.10
    let breathing = 0.10
    let system_perf = 0.10
    let monitor_freq = 0.05
    let eeg_data = 0.40
    
    let greg_multiplier = 1.2
    
    let total = (eeg_data * 0.8 + 
                keyboard_rhythm * 0.7 +
                mouse_patterns * 0.6 +
                voice_analysis * 0.7 +
                breathing * 0.8 +
                system_perf * 0.9 +
                monitor_freq * 0.7) * greg_multiplier
    
    return total
}"#;

    println!("Testing Part 4 without leading newline:");
    match parse_phi_program(part4_no_newline) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }

    // Test simplified version
    let part4_simple = r#"
function monitor_consciousness() -> Number {
    let total = 1.0
    return total
}
"#;

    println!("Testing Part 4 simplified:");
    match parse_phi_program(part4_simple) {
        Ok(_) => println!("✅ Parsed successfully\n"),
        Err(e) => println!("❌ Failed: {}\n", e),
    }
}