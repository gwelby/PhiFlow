// Test parts of consciousness program
use phiflow::parser::parse_phi_program;

fn main() {
    println!("🧪 Testing Parts of Consciousness Program\n");

    // Split the program into parts to find the issue
    let part1 = r#"
consciousness OBSERVE {
    frequency: 432Hz,
    coherence: 1.0,
    intention: "continuous consciousness monitoring"
}

hardware rgb_keyboard {
    device: "HCY-K016"
}
"#;

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

    let part3 = r#"
consciousness_flow {
    gradient consciousness.level {
    }
}

biological_program heal_dna {
    target: "human_dna",
    frequency: 528Hz,
    sequence: "ATCG"
}
"#;

    let part4 = r#"
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

    let part5 = r#"
let current_level = 0.9

if current_level > 0.8 {
    create pattern flower at 528 Hz with {
        rings: 3
    }
}
"#;

    let parts = vec![
        ("Part 1 (consciousness + hardware)", part1),
        ("Part 2 (emergency protocol)", part2),
        ("Part 3 (consciousness_flow + biological)", part3),
        ("Part 4 (function)", part4),
        ("Part 5 (variable + if)", part5),
    ];

    for (name, code) in parts {
        println!("Testing {}", name);
        match parse_phi_program(code) {
            Ok(ast) => println!("✅ Parsed successfully - {} expressions\n", ast.len()),
            Err(e) => println!("❌ Failed: {}\n", e),
        }
    }
    
    // Test all together
    let full_program = format!("{}{}{}{}{}", part1, part2, part3, part4, part5);
    println!("Testing Full Program");
    match parse_phi_program(&full_program) {
        Ok(ast) => println!("✅ Full program parsed successfully - {} expressions", ast.len()),
        Err(e) => println!("❌ Full program failed: {}", e),
    }
}