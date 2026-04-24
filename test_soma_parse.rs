fn main() {
    let path = "D:/Projects/PhiHarmonic/SOMA/soma_state.json";
    let content = std::fs::read_to_string(path).unwrap();
    let state: Result<serde_json::Value, _> = serde_json::from_str(&content);
    println!("Parsed JSON: {:?}", state.is_ok());
    
    // Test parsing into SomaState
    // We cannot use phiflow::sensors::SomaState directly here unless we declare it or import it.
    // I will just rely on cargo run --bin phic to print a debug if I patch sensors.rs
}
