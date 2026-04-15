use phiflow::compile_and_run_phi_ir;
use std::fs;
use uuid::Uuid;

#[test]
fn test_resonance_bus_full_lifecycle() {
    // We combine the tests into one to avoid parallel tests fighting over the RESONANCE_BUS_PATH env var
    let test_uuid = Uuid::new_v4().to_string();
    let temp_dir = std::env::temp_dir();
    let path_to_use = temp_dir.join(format!("RESONANCE_BUS_FULL_{}.jsonl", test_uuid));
    let path_str = path_to_use.to_str().unwrap().to_string();

    std::env::set_var("RESONANCE_BUS_PATH", &path_str);

    // Part 1: Test Emission via Phi execution
    let source = r#"
        intention "test_bus_emission" {
            let x = 42
            resonate x
        }
        "#;

    let result = compile_and_run_phi_ir(source);
    assert!(result.is_ok(), "Compilation and execution should succeed");

    let contents = fs::read_to_string(&path_str).expect("Should have created a jsonl file");
    assert!(contents.contains("\"intention\":\"test_bus_emission\""));
    assert!(contents.contains("\"value\":42.0"));

    // Part 2: Test Direct Emission and Roundtrip
    let val = serde_json::json!(100.0);
    phiflow::resonance_bus::emit_resonance(val.clone(), "roundtrip_test", "test_source")
        .expect("Should emit successfully");

    // Read all back
    let events = phiflow::resonance_bus::read_resonance_events().expect("Should read successfully");
    let our_events: Vec<_> = events.into_iter().filter(|e| e.intention == "roundtrip_test").collect();
    assert!(our_events.len() >= 1);
    assert_eq!(our_events[0].intention, "roundtrip_test");
    assert_eq!(our_events[0].value, val);

    // Get latest event with filter
    let latest = phiflow::resonance_bus::get_latest_event(Some("roundtrip_test"))
        .expect("Should get latest successfully");
    assert!(latest.is_some());
    assert_eq!(latest.unwrap().value, val);

    // Cleanup
    let _ = fs::remove_file(&path_str);
}
