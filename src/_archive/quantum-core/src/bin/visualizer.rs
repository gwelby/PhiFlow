use structopt::StructOpt;
use quantum_core::visualizer::Visualizer;
use std::path::PathBuf;

#[derive(Debug, StructOpt)]
#[structopt(name = "visualizer", about = "Quantum field visualizer")]
struct Opt {
    #[structopt(short, long, parse(from_os_str))]
    input: PathBuf,
    
    #[structopt(short, long)]
    width: u32,
    
    #[structopt(short, long)]
    height: u32,
}

fn main() {
    let opt = Opt::from_args();
    let visualizer = Visualizer::new(opt.width, opt.height);
    
    match visualizer.draw_field(&opt.input, &[]) {
        Ok(_) => println!("✨ Visualization complete!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}
