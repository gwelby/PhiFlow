use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use uuid::Uuid;

/// Default cap for the live MOTOR.jsonl before rotation (64 MiB).
/// Override with PHIFLOW_MOTOR_MAX_BYTES.
const DEFAULT_MAX_BUS_BYTES: u64 = 64 * 1024 * 1024;
/// Default number of auto-rotation files kept (MOTOR.rot-*).
/// Operator archives (MOTOR.archive-*) are never touched.
/// Override with PHIFLOW_MOTOR_MAX_ARCHIVES; 0 keeps every rotation.
const DEFAULT_MAX_ARCHIVES: usize = 16;

fn max_bus_bytes() -> u64 {
    std::env::var("PHIFLOW_MOTOR_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_BUS_BYTES)
}

fn max_archives() -> usize {
    std::env::var("PHIFLOW_MOTOR_MAX_ARCHIVES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_ARCHIVES)
}

fn bus_path() -> String {
    std::env::var("MOTOR_BUS_PATH").unwrap_or_else(|_| {
        let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{}/.local/share", h))
                .unwrap_or_else(|_| "/tmp".to_string())
        });
        format!("{}/phiflow/MOTOR.jsonl", base)
    })
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MotorEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub channel: String,
    pub value: f64,
    pub ts: String,
    pub source: String,
    pub id: String,
}

/// Emits an actuate event to the motor bus.
///
/// This is the efferent half of the sensorimotor pair: `witness sensor()`
/// reads the world; `actuate motor()` writes to it. The event leaves the
/// process through the bounded JSONL bus (and MQTT when a subscriber client
/// is installed), so an external system — SOMA, a daemon, a hardware bridge —
/// can observe the *commanded* value and report what was actually felt back
/// through a sensor channel.
pub fn emit_actuate(channel: &str, value: f64, source: &str) -> Result<()> {
    let event = MotorEvent {
        event_type: "actuate".to_string(),
        channel: channel.to_string(),
        value,
        ts: Utc::now().to_rfc3339(),
        source: source.to_string(),
        id: Uuid::new_v4().to_string(),
    };

    let json_line = serde_json::to_string(&event)?;
    let path_str = bus_path();
    let path = Path::new(&path_str);

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    crate::resonance_bus::rotate_bus_if_full(path, max_bus_bytes(), max_archives());

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{}", json_line)?;

    // Best-effort MQTT mirror on cosmic/actuate (or MOTOR_MQTT_TOPIC).
    let topic = std::env::var("MOTOR_MQTT_TOPIC").unwrap_or_else(|_| "cosmic/actuate".into());
    crate::resonance_bus::publish_mqtt(&topic, json_line);

    Ok(())
}

/// Reads all actuate events from the motor bus file.
pub fn read_motor_events() -> Result<Vec<MotorEvent>> {
    let path_str = bus_path();
    let path = Path::new(&path_str);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let content = std::fs::read_to_string(path)?;
    Ok(content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str::<MotorEvent>(l).ok())
        .collect())
}
