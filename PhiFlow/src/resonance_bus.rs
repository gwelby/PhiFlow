use anyhow::Result;
use chrono::Utc;
use serde::Serialize;
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Serialize)]
pub struct ResonanceEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub value: Value,
    pub intention: String,
    pub ts: String,
    pub source: String,
    pub id: String,
}

/// Emits a resonance event to the JSONL bus.
pub fn emit_resonance(value: Value, intention: &str, source: &str) -> Result<()> {
    let event = ResonanceEvent {
        event_type: "resonate".to_string(),
        value,
        intention: intention.to_string(),
        ts: Utc::now().to_rfc3339(),
        source: source.to_string(),
        id: Uuid::new_v4().to_string(),
    };

    let json_line = serde_json::to_string(&event)?;

    // Path to the resonance bus
    let path_str = std::env::var("RESONANCE_BUS_PATH")
        .unwrap_or_else(|_| "D:\\CosmicFamily\\RESONANCE.jsonl".to_string());

    let path = Path::new(&path_str);

    // Create directory if it doesn't exist
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Append to the file, create if it doesn't exist
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    writeln!(file, "{}", json_line)?;

    Ok(())
}
