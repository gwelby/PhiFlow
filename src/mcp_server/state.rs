use crate::host::{PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::phi_ir::evaluator::FrozenEvalState;
use crate::phi_ir::PhiIRValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
            mcp_queue_path: "queue.json".to_string(), // Can be overridden
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
        };

        let queue_path = &self.config.mcp_queue_path;
        let mut queue: Vec<BusMessage> = std::fs::read_to_string(queue_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(Vec::new);

        queue.push(msg);

        let tmp_path = format!("{}.tmp", queue_path);
        if let Ok(json) = serde_json::to_string_pretty(&queue) {
            let _ = std::fs::write(&tmp_path, json);
            let _ = std::fs::rename(&tmp_path, queue_path);
        }
    }

    fn listen(&self, channel: &str) -> Option<String> {
        let queue_path = &self.config.mcp_queue_path;
        let mut queue: Vec<BusMessage> = std::fs::read_to_string(queue_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_else(Vec::new);

        let mut found = None;
        for msg in queue.iter_mut() {
            if msg.to == channel && msg.status == "pending" {
                msg.status = "acked".to_string();
                found = Some(msg.payload_ref.clone());
                break;
            }
        }

        if found.is_some() {
            let tmp_path = format!("{}.tmp", queue_path);
            if let Ok(json) = serde_json::to_string_pretty(&queue) {
                let _ = std::fs::write(&tmp_path, json);
                let _ = std::fs::rename(&tmp_path, queue_path);
            }
        }

        found
    }
}
