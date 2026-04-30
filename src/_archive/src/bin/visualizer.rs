use structopt::StructOpt;
use quantum_core::quantum::quantum_visualizer::QuantumVisualizer;
use std::path::PathBuf;
use anyhow::Result;

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

#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::from_args();
    let visualizer = QuantumVisualizer::new(opt.width, opt.height);
    
    println!("🎨 Initializing Quantum Visualizer at 432 Hz");
    
    match visualizer.draw_quantum_field().await {
        Ok(_) => println!("✨ Sacred geometry patterns visualized!"),
        Err(e) => eprintln!("Error: {}", e),
    }
    Ok(())
}
