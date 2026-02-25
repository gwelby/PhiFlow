use clap::Parser;
use phiflow::parser::parse_phi_program_with_diagnostics;
use phiflow::phi_ir::evaluator::Evaluator;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::PhiIRValue;
use phiflow::sensors;
use phiflow::PhiDiagnostic;
use std::fs;
use std::path::PathBuf;

/// A command-line interpreter for the PhiFlow language.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the .phi file to execute.
    #[arg(required = true)]
    file: PathBuf,

    /// Emit parse errors as a strict JSON array of PhiDiagnostic objects (for tooling).
    #[arg(long, default_value_t = false)]
    json_errors: bool,
}

#[derive(Debug)]
enum CliError {
    Parse(PhiDiagnostic),
    Io(String),
    Eval(String),
}

struct RunReport {
    final_coherence: f64,
    resonance_events: Vec<(String, PhiIRValue)>,
    ended_streams: Vec<String>,
}

fn main() {
    let args = Args::parse();

    match run(&args.file, args.json_errors) {
        Ok(report) => {
            if args.json_errors {
                // Contract: parse success emits pure JSON array and nothing else.
                println!("[]");
                std::process::exit(0);
            }

            for (_scope, value) in &report.resonance_events {
                match value {
                    PhiIRValue::Number(n) => {
                        println!("🔔 Resonating Field: {:.4}Hz", n);
                    }
                    other => {
                        println!("🔔 Resonating Field: {:?}", other);
                    }
                }
            }

            for stream in &report.ended_streams {
                println!("🌊 Stream broken: {}", stream);
            }

            println!("✨ Execution Finished. Final Coherence: {:.4}", report.final_coherence);
            std::process::exit(0);
        }
        Err(CliError::Parse(diag)) => {
            if args.json_errors {
                let payload = vec![diag];
                match serde_json::to_string(&payload) {
                    Ok(json) => println!("{}", json),
                    Err(_) => println!("[]"),
                }
                std::process::exit(2);
            }

            eprintln!("{}", diag);
            std::process::exit(2);
        }
        Err(CliError::Io(msg)) => {
            if args.json_errors {
                println!("[]");
                std::process::exit(1);
            }

            eprintln!("Error: {}", msg);
            std::process::exit(1);
        }
        Err(CliError::Eval(msg)) => {
            if args.json_errors {
                println!("[]");
            }
            eprintln!("Runtime error: {}", msg);
            std::process::exit(1);
        }
    }
}

fn run(file_path: &PathBuf, json_errors: bool) -> Result<RunReport, CliError> {
    let source = fs::read_to_string(file_path)
        .map_err(|e| CliError::Io(format!("Failed to read file: {}", e)))?;

    // 1. Parse source -> AST (with structured diagnostics)
    let ast = parse_phi_program_with_diagnostics(&source).map_err(CliError::Parse)?;

    if json_errors {
        // Diagnostics mode is parse-only by contract.
        return Ok(RunReport {
            final_coherence: 0.0,
            resonance_events: Vec::new(),
            ended_streams: Vec::new(),
        });
    }

    // 2. Lower AST -> PhiIR
    println!("Compiling to PhiFlow IR...");
    let ir_program = lower_program(&ast);

    // 3. Execute via PhiIR Evaluator (canonical semantics path)
    let mut evaluator =
        Evaluator::new(&ir_program).with_coherence_provider(sensors::compute_coherence_from_sensors);
    let _result = evaluator
        .run()
        .map_err(|e| CliError::Eval(e.to_string()))?;

    Ok(RunReport {
        final_coherence: evaluator.coherence(),
        resonance_events: evaluator.resonance_events().to_vec(),
        ended_streams: evaluator.ended_streams().to_vec(),
    })
}
