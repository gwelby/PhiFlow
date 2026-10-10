//! Bounded resonance bus: the live RESONANCE.jsonl must not grow without
//! limit. A full file rotates to RESONANCE.rot-*, operator RESONANCE.archive-*
//! files are never pruned, and rotation count honors the configured bound.

use serde_json::json;

#[test]
fn test_resonance_bus_rotates_at_cap() {
    let dir = tempfile::tempdir().unwrap();
    let bus = dir.path().join("RESONANCE.jsonl");

    // Seed a live file over the cap and a decoy operator archive
    std::fs::write(&bus, "x".repeat(2048)).unwrap();
    let operator_archive = dir.path().join("RESONANCE.archive-2026-10-09.jsonl");
    std::fs::write(&operator_archive, "operator archive - do not touch").unwrap();

    std::env::set_var("RESONANCE_BUS_PATH", &bus);
    std::env::set_var("PHIFLOW_RESONANCE_MAX_BYTES", "1024");
    std::env::set_var("PHIFLOW_RESONANCE_MAX_ARCHIVES", "8");

    phiflow::resonance_bus::emit_resonance(json!(1.0), "test", "rotation-test").unwrap();

    let live = std::fs::metadata(&bus).unwrap().len();
    assert!(live < 1024, "live file should restart under cap, got {}", live);

    let rotations: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("RESONANCE.rot-"))
        .collect();
    assert_eq!(rotations.len(), 1, "expected exactly one rotation file");

    assert_eq!(
        std::fs::read_to_string(&operator_archive).unwrap(),
        "operator archive - do not touch",
        "operator archive must survive auto-rotation"
    );

    // Second emit under cap: no new rotation
    phiflow::resonance_bus::emit_resonance(json!(2.0), "test", "rotation-test").unwrap();
    let rotations2 = std::fs::read_dir(dir.path())
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("RESONANCE.rot-"))
        .count();
    assert_eq!(rotations2, 1);

    std::env::remove_var("RESONANCE_BUS_PATH");
    std::env::remove_var("PHIFLOW_RESONANCE_MAX_BYTES");
    std::env::remove_var("PHIFLOW_RESONANCE_MAX_ARCHIVES");
}
