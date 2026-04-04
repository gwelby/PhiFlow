use clap::Parser;
use phiflow::phi_ir::{emitter, vm::PhiVm, PhiIRValue};
use std::fs;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Execute standalone .phivm bytecode")]
struct Args {
    /// Input PhiVM bytecode file.
    #[arg(required = true)]
    input: PathBuf,

    /// Print bytecode summary before execution.
    #[arg(long, default_value_t = false)]
    disassemble: bool,

    /// Print the final VM value stack after execution.
    #[arg(long, default_value_t = false)]
    dump_stack: bool,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("{}", err);
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), String> {
    let bytes = fs::read(&args.input)
        .map_err(|e| format!("Failed to read bytecode {}: {}", args.input.display(), e))?;

    if args.disassemble {
        println!("{}", emitter::disassemble(&bytes));
    }

    let mut vm = PhiVm::from_bytes(&bytes)
        .map_err(|e| format!("Failed to load bytecode {}: {}", args.input.display(), e))?;
    let result = vm
        .run()
        .map_err(|e| format!("VM runtime error in {}: {}", args.input.display(), e))?;

    if args.dump_stack {
        println!("result: {}", render_value(&result, vm.string_table())?);
        for (index, value) in vm.value_stack().iter().enumerate() {
            println!(
                "stack[{index}]: {}",
                render_value(value, vm.string_table())?
            );
        }
    } else {
        println!("{}", render_value(&result, vm.string_table())?);
    }

    Ok(())
}

fn render_value(value: &PhiIRValue, string_table: &[String]) -> Result<String, String> {
    match value {
        PhiIRValue::Number(n) => Ok(n.to_string()),
        PhiIRValue::String(index) => {
            let resolved = string_table
                .get(*index as usize)
                .ok_or_else(|| format!("VM produced invalid string index {}", index))?;
            serde_json::to_string(resolved)
                .map_err(|e| format!("Failed to render string value: {}", e))
        }
        PhiIRValue::Boolean(value) => Ok(value.to_string()),
        PhiIRValue::Void => Ok("void".to_string()),
    }
}
