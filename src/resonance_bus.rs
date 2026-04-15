use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, Clone)]
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

/// Reads all resonance events from the JSONL bus file.
pub fn read_resonance_events() -> Result<Vec<ResonanceEvent>> {
    let path_str = std::env::var("RESONANCE_BUS_PATH")
        .unwrap_or_else(|_| "D:\\CosmicFamily\\RESONANCE.jsonl".to_string());
    let path = Path::new(&path_str);

    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str::<ResonanceEvent>(&line) {
            events.push(event);
        }
    }

    Ok(events)
}

/// Retrieves the latest resonance event from the bus, optionally filtered by intention.
pub fn get_latest_event(intention_filter: Option<&str>) -> Result<Option<ResonanceEvent>> {
    let events = read_resonance_events()?;

    let filtered: Vec<ResonanceEvent> = events
        .into_iter()
        .filter(|e| {
            if let Some(target) = intention_filter {
                e.intention == target
            } else {
                true
            }
        })
        .collect();

    Ok(filtered.into_iter().last())
}
