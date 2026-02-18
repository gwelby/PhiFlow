use clap::Parser;
use std::fs;
use std::path::PathBuf;
use phiflow::parser::parse_phi_program;
use phiflow::ir::lowering::lower;
use phiflow::ir::vm::PhiVm;

/// A command-line interpreter for the PhiFlow language.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to the .phi file to execute.
    #[arg(required = true)]
    file: PathBuf,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if let Err(e) = run(&args.file).await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

async fn run(file_path: &PathBuf) -> Result<(), String> {
    let source = fs::read_to_string(file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // 1. Parse Source -> AST
    let ast = parse_phi_program(&source)
        .map_err(|e| format!("Parse Error: {:?}", e))?;

    // 2. Lower AST -> IR
    println!("Compiling to PhiFlow IR...");
    let ir_program = lower(&ast);

    // 3. Execute IR on PhiVm
    let mut vm = PhiVm::new();
    vm.run(&ir_program).await;

    Ok(())
}