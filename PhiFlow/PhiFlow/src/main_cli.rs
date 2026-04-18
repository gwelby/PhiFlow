use clap::Parser;
use std::fs;
use std::path::PathBuf;

mod phi_core;
mod parser;
mod interpreter;
mod visualization;

use parser::parse_phi_program;
use interpreter::PhiInterpreter;
use crate::parser::PhiValue;

/// A command-line interpreter for the PhiFlow language.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the .phi file to execute.
    #[arg(required = true)]
    file: PathBuf,
}

fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args.file) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(file_path: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let expressions = parse_phi_program(&source)?;

    let mut interpreter = PhiInterpreter::new();
    let result = interpreter.execute(expressions)?;

    if result != PhiValue::Void {
        println!("{:?}", result);
    }

    Ok(())
}