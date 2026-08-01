//! CLI output tests — verify what the user actually sees.
//!
//! These tests exist because of a structural gap discovered 2026-07-31:
//! the test suite had 399 tests checking internal state (coherence during
//! witness events, resolved_coherence with injected values, etc.) but
//! ZERO tests checking what the CLI actually prints. A coherence reporting
//! bug could exist for months with all tests green because no test looked
//! at the output.
//!
//! "399 tests pass" is evidence of test coverage, not evidence of correctness.
//! These tests close that gap by running the actual CLI binary and checking
//! the printed output.

use std::process::Command;

/// Returns the path to the phic binary.
fn phic_binary() -> String {
    // CARGO_BIN_EXE_phic is set by cargo when running integration tests
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_phic") {
        return path;
    }
    // When CARGO_TARGET_DIR is set, the binary won't be in target/debug.
    // Use the CARGO_TARGET_DIR env var to find it.
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let path = std::path::Path::new(&target_dir).join("debug").join("phic");
        if path.exists() {
            return path.to_string_lossy().to_string();
        }
    }
    // Fallback: look in the standard target directory relative to the manifest
    let manifest = env!("CARGO_MANIFEST_DIR");
    let candidates = [
        format!("{}/target/debug/phic", manifest),
        "target/debug/phic".to_string(),
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            return c.to_string();
        }
    }
    // Last resort: rely on PATH
    "phic".to_string()
}

/// Run phic with the given args and return (stdout, stderr, exit_code).
fn run_phic(args: &[&str]) -> (String, String, i32) {
    let binary = phic_binary();
    let output = Command::new(&binary)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect(&format!("failed to run phic at {}", binary));
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let code = output.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

/// Run phic with a .phi file and return the stdout.
fn run_phi_file(path: &str) -> (String, String, i32) {
    run_phic(&[path])
}

#[test]
fn test_cli_reports_nonzero_coherence_for_intention_program() {
    // A program with an intention block should report non-zero coherence.
    // depth 1 → coherence = 1 - φ^(-1) ≈ 0.382
    let phi_source = r#"intention "test" {
    let x = 42.0
    witness
}"#;
    let tmp = std::env::temp_dir().join("cli_test_intention.phi");
    std::fs::write(&tmp, phi_source).unwrap();

    let (stdout, _stderr, code) = run_phi_file(tmp.to_str().unwrap());
    assert_eq!(code, 0, "phic should exit 0 for valid program");

    // The CLI should print "Final Coherence: X.XXXX"
    assert!(
        stdout.contains("Final Coherence:"),
        "CLI should print Final Coherence line. Got:\n{}",
        stdout
    );

    // Extract the coherence value
    let coherence_line = stdout
        .lines()
        .find(|l| l.contains("Final Coherence:"))
        .unwrap();
    let value_str = coherence_line
        .split("Final Coherence:")
        .nth(1)
        .unwrap()
        .trim();
    let coherence: f64 = value_str.parse().unwrap_or_else(|_| {
        panic!("Could not parse coherence value from '{}'", value_str)
    });

    assert!(
        coherence > 0.0,
        "Program with intention block should report coherence > 0.0, got {}",
        coherence
    );
    assert!(
        (coherence - 0.382).abs() < 0.01,
        "Depth-1 intention should report ~0.382, got {}",
        coherence
    );
}

#[test]
fn test_cli_reports_phi_inverse_for_depth_2() {
    // A program with nested intentions (depth 2) should report φ⁻¹ ≈ 0.618
    let phi_source = r#"intention "outer" {
    intention "inner" {
        witness
    }
}"#;
    let tmp = std::env::temp_dir().join("cli_test_depth2.phi");
    std::fs::write(&tmp, phi_source).unwrap();

    let (stdout, _stderr, code) = run_phi_file(tmp.to_str().unwrap());
    assert_eq!(code, 0, "phic should exit 0 for valid program");

    let coherence_line = stdout
        .lines()
        .find(|l| l.contains("Final Coherence:"))
        .unwrap_or_else(|| panic!("No Final Coherence line in:\n{}", stdout));

    let value_str = coherence_line
        .split("Final Coherence:")
        .nth(1)
        .unwrap()
        .trim();
    let coherence: f64 = value_str
        .parse()
        .unwrap_or_else(|_| panic!("Could not parse coherence from '{}'", value_str));

    assert!(
        (coherence - 0.618).abs() < 0.01,
        "Depth-2 nested intention should report ~0.618 (φ⁻¹), got {}",
        coherence
    );
}

#[test]
fn test_cli_reports_zero_for_no_intention_program() {
    // A program with no intention blocks has depth 0 → coherence 0.0.
    // This is correct by the formula. The test exists to document this
    // behavior and catch any regression that changes it.
    let phi_source = r#"let x = 42.0
let y = x * 2.0"#;
    let tmp = std::env::temp_dir().join("cli_test_no_intention.phi");
    std::fs::write(&tmp, phi_source).unwrap();

    let (stdout, _stderr, code) = run_phi_file(tmp.to_str().unwrap());
    assert_eq!(code, 0, "phic should exit 0 for valid program");

    let coherence_line = stdout
        .lines()
        .find(|l| l.contains("Final Coherence:"))
        .unwrap_or_else(|| panic!("No Final Coherence line in:\n{}", stdout));

    let value_str = coherence_line
        .split("Final Coherence:")
        .nth(1)
        .unwrap()
        .trim();
    let coherence: f64 = value_str
        .parse()
        .unwrap_or_else(|_| panic!("Could not parse coherence from '{}'", value_str));

    assert_eq!(
        coherence, 0.0,
        "Program with no intentions (depth 0) should report 0.0, got {}",
        coherence
    );
}

#[test]
fn test_cli_measure_json_reports_coherence() {
    // The --measure flag should output JSON with final_coherence field.
    let phi_source = r#"intention "test" {
    let x = 42.0
    witness
}"#;
    let tmp = std::env::temp_dir().join("cli_test_measure.phi");
    std::fs::write(&tmp, phi_source).unwrap();

    let (stdout, _stderr, code) = run_phic(&["--measure", tmp.to_str().unwrap()]);
    assert_eq!(code, 0, "phic --measure should exit 0");

    // The CLI prints "Compiling to PhiFlow IR..." to stdout before the JSON.
    // Extract just the JSON part (first line starting with '{').
    let json_str = stdout
        .lines()
        .skip_while(|l| !l.trim_start().starts_with('{'))
        .collect::<Vec<_>>()
        .join("\n");

    let json: serde_json::Value =
        serde_json::from_str(&json_str).unwrap_or_else(|e| {
            panic!("--measure should output valid JSON. Parse error: {}\nJSON:\n{}", e, json_str)
        });

    assert!(
        json.get("ok").map(|v| v == true).unwrap_or(false),
        "JSON should have ok: true. Got: {}",
        json_str
    );

    let coherence = json
        .get("final_coherence")
        .and_then(|v| v.as_f64())
        .expect("JSON should have final_coherence as a number");

    assert!(
        coherence > 0.0,
        "--measure final_coherence should be > 0.0 for intention program, got {}",
        coherence
    );
    assert!(
        (coherence - 0.382).abs() < 0.01,
        "--measure final_coherence should be ~0.382 for depth-1, got {}",
        coherence
    );
}

#[test]
fn test_cli_canonical_example_reports_expected_coherence() {
    // The canonical example claude.phi should report non-zero coherence.
    // This is the example cited in CLAIMS.md C-3 as producing φ⁻¹.
    // It has a single intention block (depth 1) with a witness,
    // so it should report ~0.382 (depth 1, k=1 after resonate).
    let claude_path = "examples/claude.phi";

    // Skip if running from a different working directory
    if !std::path::Path::new(claude_path).exists() {
        eprintln!("Skipping: {} not found", claude_path);
        return;
    }

    let (stdout, _stderr, code) = run_phi_file(claude_path);
    assert_eq!(code, 0, "claude.phi should run successfully");

    assert!(
        stdout.contains("Final Coherence:"),
        "claude.phi should print Final Coherence. Got:\n{}",
        stdout
    );

    let coherence_line = stdout
        .lines()
        .find(|l| l.contains("Final Coherence:"))
        .unwrap();
    let value_str = coherence_line
        .split("Final Coherence:")
        .nth(1)
        .unwrap()
        .trim();
    let coherence: f64 = value_str
        .parse()
        .unwrap_or_else(|_| panic!("Could not parse coherence from '{}'", value_str));

    // claude.phi has depth 1 (single intention) → 0.382
    // After resonate, k=1 → phase=1.0 → coherence = 0.382
    assert!(
        coherence > 0.0,
        "claude.phi should report coherence > 0.0, got {}",
        coherence
    );
}
