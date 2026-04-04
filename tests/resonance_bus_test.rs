use phiflow::compile_and_run_phi_ir;
use std::fs;
use std::path::Path;
use uuid::Uuid;

#[test]
fn test_resonance_bus_emission() {
    // We override the path to a temporary file for the test
    let test_uuid = Uuid::new_v4().to_string();
    let test_path = format!("D:\\CosmicFamily\\RESONANCE_TEST_{}.jsonl", test_uuid);

    // Fallback if D: isn't writable or we're in CI
    let temp_dir = std::env::temp_dir();
    let fallback_path = temp_dir.join(format!("RESONANCE_TEST_{}.jsonl", test_uuid));
    let path_to_use = if Path::new("D:\\CosmicFamily").exists() {
        test_path
    } else {
        fallback_path.to_str().unwrap().to_string()
    };

    std::env::set_var("RESONANCE_BUS_PATH", &path_to_use);

    let source = r#"
        intention "test_bus_emission" {
            let x = 42
            resonate x
        }
        "#;

    let result = compile_and_run_phi_ir(source);
    assert!(result.is_ok(), "Compilation and execution should succeed");

    let contents = fs::read_to_string(&path_to_use).expect("Should have created a jsonl file");

    println!("File contents:\n{}", contents);
    assert!(contents.contains("\"event_type\":\"resonate\""));
    assert!(contents.contains("\"intention\":\"test_bus_emission\""));
    assert!(contents.contains("\"value\":42.0"));

    // Cleanup
    let _ = fs::remove_file(&path_to_use);
}
