//! coherence_report — the intention-vs-result coherence playground.
//!
//! Accepts a `.phi` snippet, runs it through the existing parser / lowerer /
//! evaluator pipeline, and prints a short plain-English report of how aligned
//! the run was with the stated intention(s) — alongside the raw coherence
//! number the runtime already produces.
//!
//! Usage:
//!     coherence_report [--timeline] <path-to.phi>
//!
//! Try the bundled snippets:
//!     coherence_report examples/coherence_playground/aligned.phi
//!     coherence_report examples/coherence_playground/drifts.phi
//!     coherence_report examples/coherence_playground/disconnected.phi
//!
//! Pass `--timeline` to additionally print a per-witness-checkpoint table and
//! a small sparkline of coherence over the run, which is useful when there
//! are several `witness` calls and you want to see *when* coherence shifted
//! rather than just the final verdict. The default report is unchanged.
//!
//! The tool only relies on the four core constructs the runtime already ships
//! with — `intention`, `stream`, `witness`, `resonate`, `coherence` — and adds
//! no new IR nodes or keywords.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use phiflow::parser::parse_phi_program_with_diagnostics;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program_checked;
use phiflow::phi_ir::vm_state::VmWitnessEvent;

fn main() -> ExitCode {
    let mut show_timeline = false;
    let mut path: Option<PathBuf> = None;
    for arg in env::args().skip(1) {
        match arg.as_str() {
            "--timeline" => show_timeline = true,
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other if other.starts_with("--") => {
                eprintln!("coherence_report: unknown flag '{}'", other);
                eprintln!();
                print_usage();
                return ExitCode::from(2);
            }
            other => {
                if path.is_some() {
                    eprintln!(
                        "coherence_report: only one input path is supported (got an extra '{}')",
                        other
                    );
                    return ExitCode::from(2);
                }
                path = Some(PathBuf::from(other));
            }
        }
    }
    let path = match path {
        Some(p) => p,
        None => {
            print_usage();
            return ExitCode::from(2);
        }
    };

    let source = match fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Could not read {}: {}", path.display(), e);
            return ExitCode::from(1);
        }
    };

    let intentions = collect_declared_intentions(&source);

    let ast = match parse_phi_program_with_diagnostics(&source) {
        Ok(ast) => ast,
        Err(diag) => {
            print_header(&path, &intentions);
            println!();
            println!("This snippet did not parse, so the intention was never given a chance");
            println!("to run. The parser said:");
            println!();
            println!("  {}", diag);
            return ExitCode::from(2);
        }
    };

    let ir_program = match lower_program_checked(&ast) {
        Ok(p) => p,
        Err(e) => {
            print_header(&path, &intentions);
            println!();
            println!("This snippet parsed but could not be lowered into runnable form,");
            println!("so the intention was never given a chance to run. The lowerer said:");
            println!();
            println!("  {}", e);
            return ExitCode::from(2);
        }
    };

    let mut evaluator = Evaluator::new(ir_program);
    evaluator.max_steps = Some(1_000_000);

    if let Err(e) = evaluator.run() {
        print_header(&path, &intentions);
        println!();
        println!("The program started running but stopped before it finished. The");
        println!("runtime said:");
        println!();
        println!("  {}", e);
        return ExitCode::from(1);
    }

    let witness_log = &evaluator.witness_log;
    let resonance_events = evaluator.resonance_events();
    let ended_streams = evaluator.ended_streams();
    let post_run_coherence = evaluator.resolved_coherence();

    let peak_coherence = witness_log
        .iter()
        .map(|w| w.coherence)
        .fold(f64::NEG_INFINITY, f64::max);
    let peak_coherence = if peak_coherence.is_finite() {
        Some(peak_coherence)
    } else {
        None
    };
    let first_witness = witness_log.first().map(|w| w.coherence);
    let last_witness = witness_log.last().map(|w| w.coherence);

    print_header(&path, &intentions);
    println!();
    print_report(
        &intentions,
        witness_log.len(),
        resonance_events.len(),
        ended_streams,
        peak_coherence,
        first_witness,
        last_witness,
        post_run_coherence,
    );

    if show_timeline {
        println!();
        print_timeline(witness_log);
    }

    ExitCode::SUCCESS
}

fn print_usage() {
    eprintln!("usage: coherence_report [--timeline] <path-to.phi>");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --timeline   Also print a per-witness-checkpoint table and a");
    eprintln!("               sparkline of coherence over the run.");
    eprintln!();
    eprintln!("Try one of the bundled snippets:");
    eprintln!("  coherence_report examples/coherence_playground/aligned.phi");
    eprintln!("  coherence_report examples/coherence_playground/drifts.phi");
    eprintln!("  coherence_report examples/coherence_playground/disconnected.phi");
}

fn print_header(path: &PathBuf, intentions: &[String]) {
    println!("Coherence report for {}", path.display());
    println!("------------------------------------------------------------");
    if intentions.is_empty() {
        println!("Stated intention: (none — the snippet has no `intention` block)");
    } else if intentions.len() == 1 {
        println!("Stated intention: \"{}\"", intentions[0]);
    } else {
        println!("Stated intentions:");
        for name in intentions {
            println!("  - \"{}\"", name);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    intentions: &[String],
    witness_count: usize,
    resonance_count: usize,
    ended_streams: &[String],
    peak_coherence: Option<f64>,
    first_witness: Option<f64>,
    last_witness: Option<f64>,
    post_run_coherence: f64,
) {
    println!("What the runtime measured");
    println!("  witness checkpoints : {}", witness_count);
    println!("  resonance events    : {}", resonance_count);
    println!("  streams completed   : {}", ended_streams.len());
    if let Some(peak) = peak_coherence {
        println!("  peak coherence      : {:.4}", peak);
    } else {
        println!("  peak coherence      : (no `witness` was reached)");
    }
    if let (Some(first), Some(last)) = (first_witness, last_witness) {
        if witness_count > 1 {
            println!(
                "  first → last witness: {:.4} → {:.4}",
                first, last
            );
        }
    }
    // The post-run coherence is what `phic` prints as "Final Coherence". It is
    // measured *after* every intention/stream has been popped, so for any
    // single-intention snippet it is always 0.0000. We still surface it so the
    // raw number lines up with what the existing runner prints.
    println!("  post-run coherence  : {:.4}", post_run_coherence);
    println!();

    println!("Plain-English reading");
    let verdict = describe_run(
        intentions,
        witness_count,
        resonance_count,
        peak_coherence,
        first_witness,
        last_witness,
    );
    for line in verdict.lines() {
        println!("  {}", line);
    }
}

/// Categorise the run into a short, plain-language paragraph that a
/// non-programmer can read. The verdict is built from the same four signals
/// the runtime already produces — coherence, witnesses, resonances, and
/// whether any intention was ever entered — and never invents new metrics.
fn describe_run(
    intentions: &[String],
    witness_count: usize,
    resonance_count: usize,
    peak_coherence: Option<f64>,
    first_witness: Option<f64>,
    last_witness: Option<f64>,
) -> String {
    if intentions.is_empty() {
        return "The snippet never declared an intention, so the runtime had nothing to \
align against. Coherence stayed at 0.00 for the whole run. Wrap your logic in \
an `intention \"name\" { ... }` block and re-run to get a real reading."
            .to_string();
    }

    let intention_label = if intentions.len() == 1 {
        format!("intention \"{}\"", intentions[0])
    } else {
        format!(
            "{} nested intentions (innermost: \"{}\")",
            intentions.len(),
            intentions.last().expect("non-empty checked above")
        )
    };

    let peak = match peak_coherence {
        Some(p) => p,
        None => {
            return format!(
                "The {} ran, but the snippet never called `witness`, so the runtime \
never paused to compare what it was doing against what it intended. Add at \
least one `witness` line inside the intention to get a coherence reading.",
                intention_label
            );
        }
    };

    let band = match peak {
        p if p >= 0.85 => "fully aligned",
        p if p >= 0.60 => "strongly aligned",
        p if p >= 0.35 => "loosely aligned",
        p if p > 0.0 => "barely aligned",
        _ => "not aligned at all",
    };

    // Compare the very first and very last witness readings — both taken
    // *while* the intention was still in scope. This is the meaningful
    // "did the run hold its alignment?" signal. Post-run coherence isn't
    // useful here because every intention pop drops it back to 0.
    let drift = match (first_witness, last_witness) {
        (Some(first), Some(last)) if witness_count > 1 => Some((first, last, first - last)),
        _ => None,
    };
    let drifted_significantly = matches!(drift, Some((_, _, d)) if d > 0.05);

    let mut summary = format!(
        "The {} reached a peak coherence of {:.2} ({}). ",
        intention_label, peak, band
    );

    if drifted_significantly {
        // Don't claim the run "stayed close to its purpose" if we're about to
        // describe a drop — let the drift sentence carry the story.
        summary.push_str(
            "The intention was reached at the start of the run, but then slipped: ",
        );
    } else if peak >= 0.60 {
        summary.push_str(
            "The run stayed close to its stated purpose — the witness checkpoints \
saw a focused intention with little phase decay from competing resonances. ",
        );
    } else if peak >= 0.35 {
        summary.push_str(
            "The run was on-purpose but unfocused — the runtime saw the intention, \
but enough competing resonances were sharing the field that alignment was \
diluted. ",
        );
    } else if peak > 0.0 {
        summary.push_str(
            "The run barely held its purpose — either the intention was very shallow \
or so many things were resonating at once that the signal washed out. ",
        );
    } else {
        summary.push_str(
            "The run never actually reached its purpose — every witness checkpoint \
saw zero alignment, even though an intention block was declared. ",
        );
    }

    if let Some((first, last, d)) = drift {
        if d > 0.05 {
            summary.push_str(&format!(
                "the first witness saw {:.2}, but the last witness saw only {:.2} as \
unrelated resonances piled into the same scope.",
                first, last
            ));
        } else if d < -0.05 {
            summary.push_str(&format!(
                "Coherence actually improved across the run: the first witness saw \
{:.2} and the last saw {:.2}.",
                first, last
            ));
        } else {
            summary.push_str(
                "Coherence held steady across all witness checkpoints — the run \
neither tightened nor drifted.",
            );
        }
    } else if resonance_count >= 6 && peak < 0.60 {
        summary.push_str(&format!(
            "({} resonance events fired in this run; past about 2–3 in the same scope, \
phase decay starts pulling the coherence number down.)",
            resonance_count
        ));
    }

    summary.trim_end().to_string()
}

/// Scan the source text for `intention "name" {` declarations.
///
/// This is intentionally a tolerant best-effort text scan rather than an AST
/// walk: it lets the playground name the intention even when later parts of
/// the snippet fail to parse or lower (which is exactly the case we most want
/// a friendly report for). The trade-off is that it only recognises the
/// canonical form `intention "name" { ... }` at the start of a line. Edge
/// formatting (e.g. an `intention "name"` token sitting inside a `/* ... */`
/// block comment, on the same line as another statement, or with the keyword
/// preceded by non-whitespace characters) will not be picked up; the report
/// will then just omit the name and fall back to "(none)" or count by depth.
/// This is acceptable because the numeric coherence reading still comes from
/// the real evaluator.
fn collect_declared_intentions(source: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("intention") {
            let rest = rest.trim_start();
            if let Some(after_open) = rest.strip_prefix('"') {
                if let Some(end) = after_open.find('"') {
                    let name = &after_open[..end];
                    if !name.is_empty() {
                        out.push(name.to_string());
                    }
                }
            }
        }
    }
    out
}

/// Print the per-witness-checkpoint table and a sparkline of coherence over
/// the run. This is the body of the `--timeline` flag.
///
/// Each row corresponds to one entry in `evaluator.witness_log` (i.e. one
/// `witness` call that the evaluator actually reached). The columns are read
/// straight off the `VmWitnessEvent` and never invent new metrics.
pub(crate) fn print_timeline(witness_log: &[VmWitnessEvent]) {
    println!("Timeline (per-witness checkpoint)");
    if witness_log.is_empty() {
        println!("  (no witness checkpoints were reached during this run)");
        return;
    }

    let scope_strings: Vec<String> = witness_log
        .iter()
        .map(|w| format_intention_scope(&w.intention_stack))
        .collect();

    let scope_header = "intention scope";
    let scope_width = scope_strings
        .iter()
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(0)
        .max(scope_header.chars().count());

    let idx_width = witness_log.len().to_string().len().max(3);

    println!(
        "  {:>idx_w$}  {:<scope_w$}  {:>9}  {:>10}",
        "idx",
        scope_header,
        "coherence",
        "resonances",
        idx_w = idx_width,
        scope_w = scope_width,
    );
    println!(
        "  {} {} {} {}",
        "-".repeat(idx_width),
        "-".repeat(scope_width + 1),
        "-".repeat(9),
        "-".repeat(10),
    );
    for (i, (event, scope)) in witness_log.iter().zip(scope_strings.iter()).enumerate() {
        println!(
            "  {:>idx_w$}  {:<scope_w$}  {:>9.4}  {:>10}",
            i + 1,
            scope,
            event.coherence,
            event.resonance_count,
            idx_w = idx_width,
            scope_w = scope_width,
        );
    }

    let coherences: Vec<f64> = witness_log.iter().map(|w| w.coherence).collect();
    let sparkline = sparkline_for(&coherences);
    println!();
    println!("  coherence over time: {}", sparkline);
}

/// Format the active intention stack as `outer > middle > inner`.
///
/// `intention_stack` is recorded innermost-last by the evaluator, which is
/// the natural reading order ("we're inside outer, which is inside middle,
/// currently in inner"). Returns `(no intention)` when the stack is empty
/// — this can happen if `witness` is reached before any `intention` block,
/// which the runtime allows.
fn format_intention_scope(stack: &[String]) -> String {
    if stack.is_empty() {
        "(no intention)".to_string()
    } else {
        stack.join(" > ")
    }
}

/// Build a unicode-block sparkline from a series of coherence values.
///
/// Coherence is normalised to the fixed [0.0, 1.0] band the runtime defines
/// (rather than min/max within the run), so a flat run at 0.6 always renders
/// the same height and visual comparisons across runs are meaningful. Values
/// outside the band are clamped. Non-finite values render as a space.
fn sparkline_for(values: &[f64]) -> String {
    const BARS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let mut out = String::with_capacity(values.len());
    for &v in values {
        if !v.is_finite() {
            out.push(' ');
            continue;
        }
        let clamped = v.clamp(0.0, 1.0);
        // Pick a bucket in [0, BARS.len()-1]. 1.0 maps to the tallest bar,
        // 0.0 to the shortest — we never emit a blank for an in-band value
        // because that would be indistinguishable from a missing checkpoint.
        let last = BARS.len() - 1;
        let idx = (clamped * last as f64).round() as usize;
        let idx = idx.min(last);
        out.push(BARS[idx]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(scope: &[&str], coherence: f64, resonances: usize) -> VmWitnessEvent {
        VmWitnessEvent {
            intention_stack: scope.iter().map(|s| s.to_string()).collect(),
            coherence,
            register_count: 0,
            resonance_count: resonances,
            agent_name: None,
        }
    }

    #[test]
    fn sparkline_renders_one_glyph_per_value() {
        let s = sparkline_for(&[0.0, 0.5, 1.0]);
        assert_eq!(s.chars().count(), 3);
    }

    #[test]
    fn sparkline_uses_tallest_bar_at_one_and_shortest_at_zero() {
        let s = sparkline_for(&[0.0, 1.0]);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[1], '█');
    }

    #[test]
    fn sparkline_drift_drops_visibly() {
        // The "drifts" snippet's two checkpoints: ~0.618 then ~0.309.
        // The first should render strictly taller than the second.
        let s = sparkline_for(&[0.618, 0.309]);
        let chars: Vec<char> = s.chars().collect();
        let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        let pos = |c: char| bars.iter().position(|b| *b == c).unwrap();
        assert!(
            pos(chars[0]) > pos(chars[1]),
            "expected first bar taller than second, got {:?}",
            chars
        );
    }

    #[test]
    fn sparkline_clamps_out_of_band_values() {
        let s = sparkline_for(&[-0.5, 1.5]);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[1], '█');
    }

    #[test]
    fn sparkline_handles_non_finite() {
        let s = sparkline_for(&[f64::NAN, 0.5]);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], ' ');
        // 0.5 sits between two buckets (round(0.5 * 7) = 4); we just check it
        // rendered *something* visible in the in-band range, not the exact bar.
        let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        assert!(bars.contains(&chars[1]));
    }

    #[test]
    fn intention_scope_joins_outer_to_inner() {
        assert_eq!(
            format_intention_scope(&["outer".into(), "inner".into()]),
            "outer > inner"
        );
    }

    #[test]
    fn intention_scope_handles_empty_stack() {
        assert_eq!(format_intention_scope(&[]), "(no intention)");
    }

    #[test]
    fn print_timeline_runs_for_empty_log() {
        // Smoke test: must not panic and must not divide by zero on the
        // sparkline / column-width calculations.
        print_timeline(&[]);
    }

    #[test]
    fn print_timeline_runs_for_realistic_log() {
        // Smoke test: matches the shape of the bundled `drifts.phi` run.
        let log = vec![
            ev(&["stay_with_one_signal", "follow_one_signal"], 0.618, 1),
            ev(&["stay_with_one_signal", "follow_one_signal"], 0.309, 4),
        ];
        print_timeline(&log);
    }
}
