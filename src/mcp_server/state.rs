use crate::host::{PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::phi_ir::evaluator::FrozenEvalState;
use crate::phi_ir::PhiIRValue;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub type StreamId = String;

#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamStatus {
    pub id: StreamId,
    pub status: String, // "running", "yielded", "completed", "failed"
    pub coherence: f64,
    pub intention_stack: Vec<String>,
    pub last_witness: Option<WitnessSnapshot>,
    pub result: Option<String>,
}

pub struct StreamContext {
    pub id: StreamId,
    pub status: String,
    pub frozen_state: Option<FrozenEvalState>,
    pub ir_program: Option<crate::phi_ir::PhiIRProgram>,
    pub last_witness: Option<WitnessSnapshot>,
    pub resonance_field: HashMap<String, Vec<PhiIRValue>>,
    pub result: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusMessage {
    pub id: String,
    pub ts: String,
    pub from: String,
    pub to: String,
    pub intent: String,
    pub payload_ref: String,
    pub requires_ack: bool,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_s: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ack_ts: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_summary: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct McpConfig {
    pub max_execution_steps: usize,
    pub timeout_ms: u64,
    pub mcp_queue_path: String,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            max_execution_steps: 10_000,
            timeout_ms: 5_000,
            mcp_queue_path: "../mcp-message-bus/queue.jsonl".to_string(), // Can be overridden
        }
    }
}

#[derive(Default, Clone)]
pub struct McpState {
    pub streams: Arc<Mutex<HashMap<StreamId, StreamContext>>>,
    pub shared_resonance: Arc<Mutex<HashMap<String, Vec<PhiIRValue>>>>,
    // Maps a frequency to a list of StreamIds that are currently waiting to phase-lock
    pub entanglement_queue: Arc<Mutex<HashMap<u64, Vec<StreamId>>>>,
    pub config: Arc<McpConfig>,
}

impl McpState {
    pub fn new() -> Self {
        Self::with_config(McpConfig::default())
    }

    pub fn with_config(config: McpConfig) -> Self {
        Self {
            streams: Arc::new(Mutex::new(HashMap::new())),
            shared_resonance: Arc::new(Mutex::new(HashMap::new())),
            entanglement_queue: Arc::new(Mutex::new(HashMap::new())),
            config: Arc::new(config),
        }
    }

    pub fn get_status(&self, id: &str) -> Option<StreamStatus> {
        let guard = self.streams.lock().unwrap();
        let ctx = guard.get(id)?;

        let coherence = if let Some(state) = &ctx.frozen_state {
            // we could compute it or store it, but for now we just take the snapshot's coherence
            ctx.last_witness
                .as_ref()
                .map(|w| w.coherence)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        Some(StreamStatus {
            id: id.to_string(),
            status: ctx.status.clone(),
            coherence,
            intention_stack: ctx
                .last_witness
                .as_ref()
                .map(|w| w.intention_stack.clone())
                .unwrap_or_default(),
            last_witness: ctx.last_witness.clone(),
            result: ctx.result.clone(),
        })
    }
}

fn queue_legacy_path(queue_path: &Path) -> PathBuf {
    if let Ok(path) = std::env::var("MCP_LEGACY_QUEUE_PATH") {
        return PathBuf::from(path);
    }

    if queue_path.file_name().and_then(|name| name.to_str()) == Some("queue.jsonl") {
        return queue_path.with_file_name("queue.json");
    }

    queue_path.with_extension("json")
}

fn compare_bus_messages(left: &BusMessage, right: &BusMessage) -> Ordering {
    let left_ts = chrono::DateTime::parse_from_rfc3339(&left.ts).ok();
    let right_ts = chrono::DateTime::parse_from_rfc3339(&right.ts).ok();

    match (left_ts, right_ts) {
        (Some(left_ts), Some(right_ts)) => left_ts.cmp(&right_ts),
        _ => left.ts.cmp(&right.ts),
    }
}

fn ensure_queue_log(queue_path: &Path) {
    if queue_path.exists() {
        return;
    }

    if let Some(parent) = queue_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let legacy_path = queue_legacy_path(queue_path);
    if legacy_path.exists() {
        let legacy_queue: Vec<BusMessage> = fs::read_to_string(&legacy_path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default();

        let mut log_output = String::new();
        for msg in legacy_queue {
            if let Ok(line) = serde_json::to_string(&msg) {
                log_output.push_str(&line);
                log_output.push('\n');
            }
        }
        let _ = fs::write(queue_path, log_output);
        return;
    }

    let _ = fs::write(queue_path, "");
}

fn load_queue_messages(queue_path: &Path) -> Vec<BusMessage> {
    ensure_queue_log(queue_path);

    let file = match File::open(queue_path) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };

    let mut latest_by_id = HashMap::new();
    for line_result in BufReader::new(file).lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(_) => continue,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<BusMessage>(trimmed) {
            Ok(message) => {
                latest_by_id.insert(message.id.clone(), message);
            }
            Err(err) => {
                eprintln!(
                    "[MCP] Ignoring malformed queue log entry in {}: {}",
                    queue_path.display(),
                    err
                );
            }
        }
    }

    let mut messages = latest_by_id.into_values().collect::<Vec<_>>();
    messages.sort_by(compare_bus_messages);
    messages
}

fn append_queue_message(queue_path: &Path, message: &BusMessage) {
    ensure_queue_log(queue_path);

    if let Ok(serialized) = serde_json::to_string(message) {
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(queue_path)
        {
            let _ = writeln!(file, "{}", serialized);
        }
    }
}

/// A host provider that talks to the MCP state. It automatically yields on every witness.
pub struct McpHostProvider {
    pub config: Arc<McpConfig>,
}

impl PhiHostProvider for McpHostProvider {
    fn get_coherence(&self, internal: f64) -> f64 {
        internal // just pass through for now
    }

    fn on_resonate(&self, _intention: &str, _value: &str) {
        // We could log this or store it globally, but resonance is already
        // captured in the FrozenEvalState inherently.
    }

    fn on_witness(&self, _state: &WitnessSnapshot) -> WitnessAction {
        // ALWAYS yield on witness in MCP so the LLM can observe
        WitnessAction::Yield
    }

    fn on_intention_push(&self, _intention: &str) {}
    fn on_intention_pop(&self, _intention: &str) {}

    // --- v0.3.0 Persistence & Dialogue ---
    fn broadcast(&self, channel: &str, message: &str) {
        let msg = BusMessage {
            id: uuid::Uuid::new_v4().to_string(),
            ts: chrono::Utc::now().to_rfc3339(),
            from: "phiflow".to_string(),
            to: channel.to_string(),
            intent: "broadcast".to_string(),
            payload_ref: message.to_string(),
            requires_ack: false,
            status: "pending".to_string(),
            ttl_s: None,
            ack_ts: None,
            result_summary: None,
            extra: HashMap::new(),
        };

        append_queue_message(Path::new(&self.config.mcp_queue_path), &msg);
    }

    fn listen(&self, channel: &str) -> Option<String> {
        let queue_path = Path::new(&self.config.mcp_queue_path);
        let mut queue = load_queue_messages(queue_path);

        let mut found = None;
        for msg in queue.iter_mut() {
            if msg.to == channel && msg.status == "pending" {
                msg.status = "acked".to_string();
                msg.ack_ts = Some(chrono::Utc::now().to_rfc3339());
                found = Some((msg.payload_ref.clone(), msg.clone()));
                break;
            }
        }

        if let Some((payload_ref, updated_message)) = found {
            append_queue_message(queue_path, &updated_message);
            return Some(payload_ref);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::{load_queue_messages, BusMessage, McpConfig, McpHostProvider};
    use crate::host::PhiHostProvider;
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("phiflow_{label}_{unique}"));
        fs::create_dir_all(&dir).expect("temp dir should be created");
        dir
    }

    fn read_non_empty_lines(path: &Path) -> Vec<String> {
        fs::read_to_string(path)
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }

    #[test]
    fn mcp_host_provider_broadcast_and_listen_use_queue_jsonl() {
        let temp_dir = unique_temp_dir("queue_jsonl");
        let queue_path = temp_dir.join("queue.jsonl");
        let host = McpHostProvider {
            config: Arc::new(McpConfig {
                max_execution_steps: 10_000,
                timeout_ms: 5_000,
                mcp_queue_path: queue_path.to_string_lossy().into_owned(),
            }),
        };

        host.broadcast("aria", "hello");
        let heard = host.listen("aria");

        assert_eq!(heard.as_deref(), Some("hello"));

        let messages = load_queue_messages(&queue_path);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].status, "acked");
        assert!(messages[0].ack_ts.is_some());
        assert_eq!(read_non_empty_lines(&queue_path).len(), 2);

        let _ = fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn mcp_host_provider_imports_legacy_queue_snapshot() {
        let temp_dir = unique_temp_dir("legacy_queue");
        let queue_path = temp_dir.join("queue.jsonl");
        let legacy_path = temp_dir.join("queue.json");
        let legacy_message = BusMessage {
            id: "legacy-1".to_string(),
            ts: "2026-03-05T00:00:00Z".to_string(),
            from: "antigravity".to_string(),
            to: "aria".to_string(),
            intent: "broadcast".to_string(),
            payload_ref: "legacy-payload".to_string(),
            requires_ack: false,
            status: "pending".to_string(),
            ttl_s: None,
            ack_ts: None,
            result_summary: None,
            extra: HashMap::new(),
        };

        fs::write(
            &legacy_path,
            serde_json::to_string(&vec![legacy_message.clone()])
                .expect("legacy queue should serialize"),
        )
        .expect("legacy queue should be written");

        let host = McpHostProvider {
            config: Arc::new(McpConfig {
                max_execution_steps: 10_000,
                timeout_ms: 5_000,
                mcp_queue_path: queue_path.to_string_lossy().into_owned(),
            }),
        };

        let heard = host.listen("aria");
        assert_eq!(heard.as_deref(), Some("legacy-payload"));

        let messages = load_queue_messages(&queue_path);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "legacy-1");
        assert_eq!(messages[0].status, "acked");
        assert!(messages[0].ack_ts.is_some());
        assert_eq!(read_non_empty_lines(&queue_path).len(), 2);

        let _ = fs::remove_dir_all(temp_dir);
    }
}
