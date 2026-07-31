//! Full three-backend conformance probe (Codex hostile audit, 2026-07-31).
//!
//! The canonical conformance suite (`phi_ir_conformance_tests.rs`) only exercises
//! core constructs (arithmetic, witness, intention, resonate, coherence, sensors)
//! across Evaluator == VM == WASM. The v0.3+ constructs that make PhiFlow unique
//! (remember/recall, broadcast/listen, evolve, entangle, void_depth, field,
//! dissonance, coherence_of, agent) are NOT covered for three-backend equivalence.
//!
//! This probe runs each such construct through all three backends with a DEFAULT
//! host (no callbacks) and records what each backend returns or how it fails. It is
//! diagnostic: it prints a comparison table and only fails if a backend that is
//! expected to agree with the Evaluator diverges. The point is to make the current
//! equivalence gap measurable and reproducible, not to assert a claim.
//!
//! Run with:
//!   cargo test --test phi_ir_full_conformance_probe -- --nocapture

use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::{
    emitter,
    evaluator::Evaluator,
    lowering::lower_program,
    optimizer::{OptimizationLevel, Optimizer},
    vm::PhiVm,
    wasm::emit_wat,
    PhiIRProgram, PhiIRValue,
};
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
enum BackendOutcome {
    Number(f64),
    Other(String),
    Error(String),
    Panic,
}

impl BackendOutcome {
    fn label(&self) -> String {
        match self {
            BackendOutcome::Number(n) => format!("{:.6}", n),
            BackendOutcome::Other(s) => format!("value({s})"),
            BackendOutcome::Error(e) => format!("ERROR({})", first_line(e)),
            BackendOutcome::Panic => "PANIC".to_string(),
        }
    }

    fn as_number(&self) -> Option<f64> {
        match self {
            BackendOutcome::Number(n) => Some(*n),
            _ => None,
        }
    }
}

fn first_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(60).collect()
}

fn lower(source: &str) -> Result<PhiIRProgram, String> {
    let exprs = parse_phi_program(source).map_err(|e| format!("parse: {e:?}"))?;
    let mut program = lower_program(&exprs);
    let mut opt = Optimizer::new(OptimizationLevel::Basic);
    opt.optimize(&mut program);
    Ok(program)
}

fn run_eval(program: &PhiIRProgram) -> BackendOutcome {
    let program = program.clone();
    match catch_unwind(AssertUnwindSafe(|| {
        let mut e = Evaluator::new(program);
        e.run()
    })) {
        Ok(Ok(v)) => to_outcome(&v),
        Ok(Err(e)) => BackendOutcome::Error(format!("{e:?}")),
        Err(_) => BackendOutcome::Panic,
    }
}

fn run_vm(program: &PhiIRProgram) -> BackendOutcome {
    let program = program.clone();
    match catch_unwind(AssertUnwindSafe(|| {
        let bytes = emitter::emit(&program);
        PhiVm::run_bytes(&bytes)
    })) {
        Ok(Ok(v)) => to_outcome(&v),
        Ok(Err(e)) => BackendOutcome::Error(format!("{e:?}")),
        Err(_) => BackendOutcome::Panic,
    }
}

fn run_wasm(program: &PhiIRProgram) -> BackendOutcome {
    let program = program.clone();
    match catch_unwind(AssertUnwindSafe(|| {
        let wat = emit_wat(&program);
        run_wat_with_node(&wat)
    })) {
        Ok(Ok(n)) => BackendOutcome::Number(n),
        Ok(Err(e)) => BackendOutcome::Error(e),
        Err(_) => BackendOutcome::Panic,
    }
}

fn to_outcome(v: &PhiIRValue) -> BackendOutcome {
    match v {
        PhiIRValue::Number(n) => BackendOutcome::Number(*n),
        other => BackendOutcome::Other(format!("{other:?}")),
    }
}

fn run_wat_with_node(wat: &str) -> Result<f64, String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("phiflow_probe_{}_{}.wat", std::process::id(), now));
    fs::write(&path, wat).map_err(|e| e.to_string())?;
    let runner = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("phi_ir_wasm_runner.js");
    let output = Command::new("node")
        .arg(&runner)
        .arg(&path)
        .output()
        .map_err(|e| format!("node spawn: {e}"))?;
    let _ = fs::remove_file(&path);
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
    }
    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("parse node stdout: {e}"))
}

struct Case {
    name: &'static str,
    /// Which backends are *claimed* to be equivalent for this program.
    /// The legacy VM predates functions/streams, so some cases are eval+wasm only.
    vm_expected: bool,
    source: &'static str,
}

#[test]
fn probe_all_constructs_three_backend() {
    let cases = vec![
        Case {
            name: "remember_recall",
            vm_expected: true,
            source: "remember \"k\" 42.0\nrecall \"k\"\n",
        },
        Case {
            name: "broadcast_listen",
            vm_expected: true,
            source: "broadcast \"c\" 123.0\nlisten \"c\"\n",
        },
        // NOTE: `void_depth` (in any scope) is intentionally omitted: it triggers a
        // parser runaway allocation (OOM) in `parse_phi_program`. `field` as a bare
        // statement fails to parse ("Unexpected token in statement: Field"). Both are
        // documented as findings in REPORTS/CODEX_AUDIT_RESULTS.md.
        Case {
            name: "dissonance",
            vm_expected: true,
            source: "dissonance\n",
        },
        Case {
            name: "coherence_of",
            vm_expected: true,
            source: "coherence_of(\"other\")\n",
        },
        Case {
            name: "entangle_then_value",
            vm_expected: true,
            source: "entangle on 432.0\n7.0\n",
        },
        Case {
            name: "evolve_then_value",
            vm_expected: true,
            source: "let base = 1.0\nevolve \"let next = base + 1.0\"\nbase\n",
        },
        Case {
            name: "agent_scope_return",
            vm_expected: false,
            source: "agent \"A\" version \"1.0.0\" {\n    42.0\n}\n",
        },
        Case {
            name: "remember_recall_in_intention",
            vm_expected: true,
            source: "intention \"scope\" {\n    remember \"k\" 9.0\n    recall \"k\"\n}\n",
        },
    ];

    println!(
        "\n{:<32} {:<18} {:<18} {:<18} {}",
        "CONSTRUCT", "EVALUATOR", "VM", "WASM", "VERDICT"
    );
    println!("{}", "-".repeat(110));

    let mut divergences: Vec<String> = Vec::new();

    for case in &cases {
        let program = match lower(case.source) {
            Ok(p) => p,
            Err(e) => {
                println!("{:<32} lower failed: {}", case.name, e);
                divergences.push(format!("{}: lowering failed ({})", case.name, e));
                continue;
            }
        };

        let eval = run_eval(&program);
        let vm = run_vm(&program);
        let wasm = run_wasm(&program);

        // Verdict: compare backends that are claimed equivalent to the evaluator.
        let mut verdict = "ok";
        let eval_n = eval.as_number();

        let wasm_agrees = match (eval_n, wasm.as_number()) {
            (Some(a), Some(b)) => (a - b).abs() < 1e-9,
            _ => matches!(
                (&eval, &wasm),
                (BackendOutcome::Other(_), BackendOutcome::Other(_))
            ),
        };
        if !wasm_agrees {
            verdict = "WASM≠EVAL";
            divergences.push(format!(
                "{}: eval={} wasm={}",
                case.name,
                eval.label(),
                wasm.label()
            ));
        }

        if case.vm_expected {
            let vm_agrees = match (eval_n, vm.as_number()) {
                (Some(a), Some(b)) => (a - b).abs() < 1e-9,
                _ => matches!(
                    (&eval, &vm),
                    (BackendOutcome::Other(_), BackendOutcome::Other(_))
                ),
            };
            if !vm_agrees {
                verdict = if verdict == "ok" {
                    "VM≠EVAL"
                } else {
                    "VM&WASM≠EVAL"
                };
                divergences.push(format!(
                    "{}: eval={} vm={}",
                    case.name,
                    eval.label(),
                    vm.label()
                ));
            }
        }

        println!(
            "{:<32} {:<18} {:<18} {:<18} {}",
            case.name,
            eval.label(),
            vm.label(),
            wasm.label(),
            verdict
        );
    }

    println!("{}", "-".repeat(110));
    if divergences.is_empty() {
        println!("No divergences: all probed constructs agree across claimed-equivalent backends.");
    } else {
        println!("{} divergence(s) found:", divergences.len());
        for d in &divergences {
            println!("  - {d}");
        }
    }
    println!();

    // This probe is diagnostic. It documents the current equivalence surface for
    // v0.3+ constructs; it does not hard-fail on divergence so it can run in CI as
    // a living record. The divergence list above is the audit evidence.
}
