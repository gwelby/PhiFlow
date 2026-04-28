//! End-to-end tests for the `--timeline` and `--json` flags on the
//! `coherence_report` binary. These tests shell out to the real built binary
//! so they cover argument parsing, the existing report path, and the new
//! timeline / JSON sections together.

use std::process::Command;

// Used only by the JSON round-trip tests below.
#[allow(dead_code)]
mod json_schema {
    #[derive(serde::Deserialize, Debug)]
    pub struct CoherenceRunJson {
        pub stated_intentions: Vec<String>,
        pub checkpoints: Vec<CheckpointJson>,
        pub peak_coherence: Option<f64>,
        pub first_coherence: Option<f64>,
        pub last_coherence: Option<f64>,
        pub post_run_coherence: f64,
        pub resonance_event_count: usize,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct CheckpointJson {
        pub index: usize,
        pub intention_scope: String,
        pub coherence: f64,
        pub resonance_count: usize,
    }
}

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

#[test]
fn timeline_shows_resonance_rows_between_witnesses_for_drifts() {
    // drifts.phi fires one resonance before the first witness and three
    // resonances between the first and second witness.  The timeline must
    // interleave them: the first ~ row appears before witness #1 and three ~
    // rows appear between witness #1 and witness #2.
    let out = run(&[
        "--timeline",
        "examples/coherence_playground/drifts.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // At least one resonance row must appear.
    assert!(
        stdout.contains("resonate:"),
        "expected at least one resonance row in timeline output, got:\n{}",
        stdout
    );

    // Collect the order of significant lines: witness rows (contain a
    // numeric index, e.g. "  1  ") and resonance rows ("resonate:").
    let lines: Vec<&str> = stdout.lines().collect();

    // Find the line indices for witness #1, witness #2, and all resonance rows.
    let witness1_pos = lines
        .iter()
        .position(|l| l.trim_start().starts_with("1 ") && l.contains("stay_with_one_signal"))
        .expect("witness #1 row not found");
    let witness2_pos = lines
        .iter()
        .position(|l| l.trim_start().starts_with("2 ") && l.contains("stay_with_one_signal"))
        .expect("witness #2 row not found");

    // There must be resonance rows between witness #1 and witness #2.
    let resonance_between = lines[witness1_pos + 1..witness2_pos]
        .iter()
        .filter(|l| l.contains("resonate:"))
        .count();
    assert_eq!(
        resonance_between, 3,
        "expected 3 resonance rows between witness #1 and #2, found {}; output:\n{}",
        resonance_between, stdout
    );
}

#[test]
fn timeline_resonance_rows_show_value_for_drifts() {
    // drifts.phi resonates four numeric values: 432.0 (signal), then 528.0,
    // 396.0, and 285.0 (the three noisy detours). Each timeline row must
    // include the formatted value alongside the channel name.
    let out = run(&[
        "--timeline",
        "examples/coherence_playground/drifts.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // First resonance fires 432.0 — must appear formatted to four decimal places.
    assert!(
        stdout.contains("432.0000"),
        "expected resonated value '432.0000' in timeline output, got:\n{}",
        stdout
    );
    // Three noisy resonances fire 528.0, 396.0, and 285.0.
    assert!(
        stdout.contains("528.0000"),
        "expected resonated value '528.0000' in timeline output, got:\n{}",
        stdout
    );
    assert!(
        stdout.contains("396.0000"),
        "expected resonated value '396.0000' in timeline output, got:\n{}",
        stdout
    );
    assert!(
        stdout.contains("285.0000"),
        "expected resonated value '285.0000' in timeline output, got:\n{}",
        stdout
    );

    // The value must appear on the same line as the channel name.
    let value_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("resonate:") && l.contains("432.0000"))
        .collect();
    assert_eq!(
        value_lines.len(),
        1,
        "expected exactly one resonance row containing '432.0000', found {}; output:\n{}",
        value_lines.len(),
        stdout
    );
}

#[test]
fn timeline_resonance_rows_stay_within_120_chars() {
    // drifts.phi is the representative fixture with resonance rows. Every line
    // of the timeline section must fit within 120 characters so the table
    // renders correctly on a standard terminal.
    let out = run(&[
        "--timeline",
        "examples/coherence_playground/drifts.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    let in_timeline = stdout
        .lines()
        .skip_while(|l| !l.contains("Timeline (per-witness checkpoint)"))
        .skip(1);
    for line in in_timeline {
        let char_count = line.chars().count();
        assert!(
            char_count <= 120,
            "timeline line exceeds 120 chars ({} chars): {:?}",
            char_count,
            line
        );
    }
}

#[test]
fn timeline_resonance_value_column_aligns_with_coherence_column() {
    // Both the resonance value field and the witness coherence field are
    // right-aligned 9-char slots that start at the same column offset.
    // We verify alignment by checking that the right edge of the value field
    // (end of the resonance row, which has no trailing columns) equals the
    // right edge of the coherence field in the witness row (which is followed
    // by "  " and a 10-char resonances count column).
    let out = run(&[
        "--timeline",
        "examples/coherence_playground/drifts.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // Resonance rows end exactly at the right edge of the 9-char value field
    // (no trailing columns).
    let resonance_row = stdout
        .lines()
        .find(|l| l.contains("resonate:") && l.contains("432.0000"))
        .expect("resonance row with value '432.0000' not found");
    let resonance_field_right_edge = resonance_row.len(); // no trailing content

    // Witness rows have two more columns after coherence: "  " + 10-char
    // resonances count.  Stripping those gives the right edge of the
    // coherence field, which must equal resonance_field_right_edge.
    let witness_row = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("1 ") && l.contains("stay_with_one_signal"))
        .expect("witness #1 row not found");
    // Trailing: "  " (2) + resonances column width (10) = 12 chars after coherence field.
    let coherence_field_right_edge = witness_row
        .trim_end()
        .len()
        .saturating_sub(2 + 10);

    assert_eq!(
        resonance_field_right_edge, coherence_field_right_edge,
        "right edge of resonance value field ({}) does not align with \
        coherence column right edge ({})\n\
        witness row:   {:?}\n\
        resonance row: {:?}",
        resonance_field_right_edge,
        coherence_field_right_edge,
        witness_row,
        resonance_row
    );
}

// ── JSON flag tests ──────────────────────────────────────────────────────────

#[test]
fn json_flag_produces_valid_parseable_json() {
    let out = run(&["--json", "examples/coherence_playground/aligned.phi"]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let parsed: json_schema::CoherenceRunJson =
        serde_json::from_slice(&out.stdout).expect("--json output must be valid JSON");
    // aligned.phi declares exactly one intention.
    assert_eq!(
        parsed.stated_intentions.len(),
        1,
        "expected 1 stated intention, got {:?}",
        parsed.stated_intentions
    );
    // There must be at least one witness checkpoint recorded.
    assert!(
        !parsed.checkpoints.is_empty(),
        "expected at least one checkpoint in JSON output"
    );
    // Indices must be 1-based and contiguous.
    for (i, cp) in parsed.checkpoints.iter().enumerate() {
        assert_eq!(
            cp.index,
            i + 1,
            "checkpoint index out of order: expected {}, got {}",
            i + 1,
            cp.index
        );
    }
    // peak_coherence must be present and match the maximum checkpoint coherence.
    let peak = parsed.peak_coherence.expect("peak_coherence must be Some for a run with witnesses");
    let max_from_checkpoints = parsed
        .checkpoints
        .iter()
        .map(|cp| cp.coherence)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (peak - max_from_checkpoints).abs() < 1e-9,
        "peak_coherence {} does not match max checkpoint coherence {}",
        peak,
        max_from_checkpoints
    );
}

#[test]
fn json_output_does_not_contain_text_report() {
    let out = run(&["--json", "examples/coherence_playground/aligned.phi"]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The human-readable sections must not appear in JSON mode.
    assert!(
        !stdout.contains("Plain-English reading"),
        "--json output must not contain the text report section, got:\n{}",
        stdout
    );
    assert!(
        !stdout.contains("Coherence report for"),
        "--json output must not contain the text report header, got:\n{}",
        stdout
    );
}

#[test]
fn json_and_timeline_can_be_combined() {
    let out = run(&[
        "--json",
        "--timeline",
        "examples/coherence_playground/drifts.phi",
    ]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let stdout = String::from_utf8_lossy(&out.stdout);

    // The JSON block precedes the Timeline section. Extract just the JSON
    // portion (up to and including the closing `}` of the root object) so
    // `from_str` does not trip on the trailing timeline text.
    let json_end = stdout
        .rfind('}')
        .expect("stdout must contain the closing '}' of the JSON object");
    let json_part = &stdout[..=json_end];
    let _parsed: json_schema::CoherenceRunJson =
        serde_json::from_str(json_part)
            .expect("the JSON portion of combined --json --timeline output must be valid");

    // Timeline section must also be present.
    assert!(
        stdout.contains("Timeline (per-witness checkpoint)"),
        "expected Timeline section when both --json and --timeline are set, got:\n{}",
        stdout
    );
}

#[test]
fn json_drifts_has_two_checkpoints_and_correct_scope() {
    let out = run(&["--json", "examples/coherence_playground/drifts.phi"]);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    let parsed: json_schema::CoherenceRunJson =
        serde_json::from_slice(&out.stdout).expect("--json output must be valid JSON");
    // drifts.phi has exactly two `witness` calls.
    assert_eq!(
        parsed.checkpoints.len(),
        2,
        "drifts.phi must produce exactly 2 checkpoints, got {:?}",
        parsed.checkpoints.len()
    );
    // Both checkpoints must show the nested intention scope.
    for cp in &parsed.checkpoints {
        assert!(
            cp.intention_scope.contains("stay_with_one_signal"),
            "expected intention scope to contain 'stay_with_one_signal', got '{}'",
            cp.intention_scope
        );
    }
    // first_coherence and last_coherence must be populated.
    assert!(parsed.first_coherence.is_some(), "first_coherence must be Some");
    assert!(parsed.last_coherence.is_some(), "last_coherence must be Some");
    // The run drifts, so first > last.
    let first = parsed.first_coherence.unwrap();
    let last = parsed.last_coherence.unwrap();
    assert!(
        first > last,
        "drifts.phi: expected first_coherence ({}) > last_coherence ({})",
        first,
        last
    );
}
