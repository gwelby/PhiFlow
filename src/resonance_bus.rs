use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use rumqttc::{Client, MqttOptions, QoS, Event, Packet};
use std::sync::mpsc;
use std::time::Duration;
use std::thread;
use std::sync::{OnceLock, Mutex};
use uuid::Uuid;

static MQTT_CLIENT: OnceLock<Mutex<Client>> = OnceLock::new();

/// Default cap for the live RESONANCE.jsonl before rotation (64 MiB).
/// Override with PHIFLOW_RESONANCE_MAX_BYTES.
const DEFAULT_MAX_BUS_BYTES: u64 = 64 * 1024 * 1024;
/// Default number of auto-rotation files kept (RESONANCE.rot-*).
/// Operator archives (RESONANCE.archive-*) are never touched.
/// Override with PHIFLOW_RESONANCE_MAX_ARCHIVES; 0 keeps every rotation.
const DEFAULT_MAX_ARCHIVES: usize = 16;

fn max_bus_bytes() -> u64 {
    std::env::var("PHIFLOW_RESONANCE_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_BUS_BYTES)
}

fn max_archives() -> usize {
    std::env::var("PHIFLOW_RESONANCE_MAX_ARCHIVES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_ARCHIVES)
}

/// Rotates the live bus file to `RESONANCE.rot-<ts>.jsonl` when it has
/// reached the size cap, then prunes old rotations past the archive bound.
/// Best-effort: any failure falls through to a normal append.
fn rotate_bus_if_full(path: &Path) {
    let Ok(meta) = std::fs::metadata(path) else { return };
    if meta.len() < max_bus_bytes() {
        return;
    }
    let Some(parent) = path.parent() else { return };
    let stamp = Utc::now().format("%Y%m%dT%H%M%S%.3f");
    let rotated = parent.join(format!("RESONANCE.rot-{}-{}.jsonl", stamp, std::process::id()));
    if std::fs::rename(path, &rotated).is_err() {
        return;
    }
    let keep = max_archives();
    if keep == 0 {
        return;
    }
    let mut rotations: Vec<_> = std::fs::read_dir(parent)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("RESONANCE.rot-") && n.ends_with(".jsonl"))
                .collect()
        })
        .unwrap_or_default();
    rotations.sort();
    for name in rotations.iter().take(rotations.len().saturating_sub(keep)) {
        let _ = std::fs::remove_file(parent.join(name));
    }
}

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
    let path_str = std::env::var("RESONANCE_BUS_PATH").unwrap_or_else(|_| {
        let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{}/.local/share", h))
                .unwrap_or_else(|_| "/tmp".to_string())
        });
        format!("{}/phiflow/RESONANCE.jsonl", base)
    });

    let path = Path::new(&path_str);

    // Create directory if it doesn't exist
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Bound the live bus: rotate before it grows past the cap
    rotate_bus_if_full(path);

    // Append to the file, create if it doesn't exist
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;

    writeln!(file, "{}", json_line)?;

    // Fallback: also try to push to MQTT if a global client exists
    if let Some(client_mutex) = MQTT_CLIENT.get() {
        if let Ok(mut client) = client_mutex.lock() {
            let topic = std::env::var("RESONANCE_MQTT_TOPIC").unwrap_or_else(|_| "cosmic/resonance".into());
            let _ = client.publish(topic, QoS::AtMostOnce, false, json_line);
        }
    }

    Ok(())
}

/// Reads all resonance events from the JSONL bus file.
pub fn read_resonance_events() -> Result<Vec<ResonanceEvent>> {
    let path_str = std::env::var("RESONANCE_BUS_PATH").unwrap_or_else(|_| {
        let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{}/.local/share", h))
                .unwrap_or_else(|_| "/tmp".to_string())
        });
        format!("{}/phiflow/RESONANCE.jsonl", base)
    });
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

pub struct MqttConfig {
    pub host: String,
    pub port: u16,
    pub topic: String,
}

impl Default for MqttConfig {
    fn default() -> Self {
        let host = std::env::var("RESONANCE_MQTT_HOST").unwrap_or_else(|_| "127.0.0.1".into());
        let port = std::env::var("RESONANCE_MQTT_PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(1883);
        let topic = std::env::var("RESONANCE_MQTT_TOPIC").unwrap_or_else(|_| "cosmic/resonance".into());

        Self { host, port, topic }
    }
}

pub fn subscribe_resonance_mqtt(config: MqttConfig) -> Result<mpsc::Receiver<ResonanceEvent>> {
    let mut mqttoptions = MqttOptions::new(format!("phiflow-daemon-{}", Uuid::new_v4()), &config.host, config.port);
    mqttoptions.set_keep_alive(Duration::from_secs(5));

    let (mut client, mut connection) = Client::new(mqttoptions, 10);
    client.subscribe(&config.topic, QoS::AtMostOnce)?;

    // Store cloned client globally so emit_resonance can use it.
    let _ = MQTT_CLIENT.set(Mutex::new(client.clone()));

    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        for notification in connection.iter() {
            if let Ok(Event::Incoming(Packet::Publish(p))) = notification {
                if let Ok(payload) = String::from_utf8(p.payload.to_vec()) {
                    if let Ok(event) = serde_json::from_str::<ResonanceEvent>(&payload) {
                        let _ = tx.send(event);
                    }
                }
            }
        }
    });

    Ok(rx)
}
