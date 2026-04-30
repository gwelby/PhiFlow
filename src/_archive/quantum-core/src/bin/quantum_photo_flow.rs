use std::path::PathBuf;
use quantum_core::quantum::quantum_photo_flow_pure::QuantumPhotoFlow;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let photo_flow = QuantumPhotoFlow::new();
    
    println!("🌟 Starting Quantum Photo Flow");
    println!("⚡ Created QuantumPhotoFlow instance");
    
    let test_dir = PathBuf::from("test_images");
    if !test_dir.exists() {
        std::fs::create_dir(&test_dir)?;
    }
    
    let input_path = test_dir.join("test_input.jpg");
    let output_dir = test_dir.join("quantum_frames");
    
    if !input_path.exists() {
        println!("💫 No input image found at: {}", input_path.display());
        println!("🎨 Please place a test image named 'test_input.jpg' in the test_images directory");
        return Ok(());
    }
    
    let duration_secs = 5;
    let fps = 30;
    let width = 640;
    let height = 480;
    
    println!("💫 Generating quantum frames...");
    let frames = photo_flow.photo_to_quantum_frames(
        &input_path,
        &output_dir,
        duration_secs,
        fps,
        width,
        height
    ).await?;
    
    println!("∞ Generated {} quantum frames!", frames.len());
    println!("🎬 Frames saved to: {}", output_dir.display());
    
    Ok(())
}
