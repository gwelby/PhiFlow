use clap::Parser;
use phiflow::parser::parse_phi_program;
use phiflow::phi_ir::lowering::lower_program;
use phiflow::phi_ir::optimizer::{OptimizationLevel, Optimizer};
use phiflow::phi_ir::wasm::emit_wat;
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Compile a .phi file to .wat")]
struct Args {
    /// Input PhiFlow source file.
    #[arg(required = true)]
    input: PathBuf,

    /// Output WAT file path. Defaults to <input>.wat.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let source = fs::read_to_string(&args.input)
        .map_err(|e| format!("Failed to read input file {}: {}", args.input.display(), e))?;

    let expressions = parse_phi_program(&source).map_err(|e| format!("Parse error: {}", e))?;
    let mut program = lower_program(&expressions);

    let mut optimizer = Optimizer::new(OptimizationLevel::Basic);
    optimizer.optimize(&mut program);

    let wat = emit_wat(&program);
    let out_path = args.out.unwrap_or_else(|| args.input.with_extension("wat"));

    fs::write(&out_path, wat)
        .map_err(|e| format!("Failed to write WAT file {}: {}", out_path.display(), e))?;

    println!("{}", out_path.display());
    Ok(())
}
