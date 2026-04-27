//! End-to-end tests for the `--timeline` flag on the `coherence_report`
//! binary. These tests shell out to the real built binary so they cover
//! argument parsing, the existing report path, and the new timeline section
//! together.

use std::process::Command;

fn run(args: &[&str]) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_coherence_report"));
    cmd.args(args);
    cmd.output().expect("failed to run coherence_report binary")
}

#[test]
fn default_run_does_not_print_timeline_section() {
    let out = run(&["examples/coherence_playground/drifts.phi"]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("Timeline"),
        "default run should not include the Timeline section, got:\n{}",
        stdout
    );
    // The default verdict still appears.
    assert!(stdout.contains("Plain-English reading"));
}

#[test]
fn timeline_flag_adds_per_witness_table_and_sparkline() {
    let out = run(&[
        "--timeline",
        "examples/coherence_playground/drifts.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Section header is present.
    assert!(
        stdout.contains("Timeline (per-witness checkpoint)"),
        "missing Timeline header in:\n{}",
        stdout
    );
    // Column headers.
    assert!(stdout.contains("intention scope"));
    assert!(stdout.contains("coherence"));
    assert!(stdout.contains("resonances"));
    // The drifts snippet has exactly two `witness` calls — both rows must
    // appear, with the nested intention scope shown.
    assert!(
        stdout.contains("stay_with_one_signal > follow_one_signal"),
        "expected nested intention scope in timeline rows, got:\n{}",
        stdout
    );
    // A sparkline line must follow the table.
    assert!(
        stdout.contains("coherence over time:"),
        "missing sparkline label in:\n{}",
        stdout
    );
    // The sparkline must be made of unicode block glyphs (one per witness).
    let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let line = stdout
        .lines()
        .find(|l| l.contains("coherence over time:"))
        .expect("sparkline line not found");
    let glyphs = line.trim_end().chars().rev().take(2).collect::<Vec<_>>();
    assert_eq!(glyphs.len(), 2, "expected 2 sparkline glyphs, got {:?}", glyphs);
    for g in &glyphs {
        assert!(bars.contains(g), "sparkline contains non-bar char {:?}", g);
    }
    // The drifts snippet's coherence drops, so the most recent bar should be
    // shorter than the earlier one. The first .rev() char is the *last* bar.
    let pos = |c: &char| bars.iter().position(|b| b == c).unwrap();
    let last = pos(&glyphs[0]);
    let first = pos(&glyphs[1]);
    assert!(
        first > last,
        "expected coherence to visibly drop in sparkline (first={:?} last={:?})",
        glyphs[1], glyphs[0]
    );
}

#[test]
fn timeline_flag_works_when_passed_after_path() {
    let out = run(&[
        "examples/coherence_playground/aligned.phi",
        "--timeline",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Timeline (per-witness checkpoint)"));
    assert!(stdout.contains("coherence over time:"));
}

#[test]
fn timeline_handles_run_with_no_intention() {
    let out = run(&[
        "--timeline",
        "examples/coherence_playground/disconnected.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Timeline (per-witness checkpoint)"));
    // The witness ran with no intention on the stack — the table must say so
    // rather than blowing up or leaving the column blank.
    assert!(
        stdout.contains("(no intention)"),
        "expected '(no intention)' placeholder in timeline, got:\n{}",
        stdout
    );
}

#[test]
fn unknown_flag_prints_usage_and_exits_nonzero() {
    let out = run(&["--nope", "examples/coherence_playground/aligned.phi"]);
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown flag"));
    assert!(stderr.contains("--timeline"));
}
