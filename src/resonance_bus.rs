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
