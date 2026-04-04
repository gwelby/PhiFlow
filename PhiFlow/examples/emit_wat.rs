use phiflow::wasm_host::compile_source_to_wat;
use std::fs;

fn main() {
    let source = r#"
        intention universal_bridge {
            let sacred_freq = 432.0
            resonate (sacred_freq)
            let current_coherence = coherence
            witness (current_coherence)
        }
    "#;

    match compile_source_to_wat(source) {
        Ok(wat) => {
            fs::write("polyglot_hooks.wat", wat).expect("failed to write polyglot_hooks.wat");
            println!("Successfully emitted the 5 consciousness hooks into polyglot_hooks.wat");
        }
        Err(e) => eprintln!("Failed to compile to WAT: {}", e),
    }
}
