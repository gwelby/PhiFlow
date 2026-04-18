use crate::host::{PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::phi_ir::PhiIRValue;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A host provider for the PhiFlow Daemon that provides "System" level capabilities.
/// 
/// Security:
/// - File operations outside D:\CosmicFamily require an active "SYSTEM" intention.
/// - Shell execution (if enabled) requires an active "SYSTEM" intention.
pub struct SystemHostProvider {
    base_path: PathBuf,
    intention_stack: Arc<Mutex<Vec<String>>>,
}

impl SystemHostProvider {
    pub fn new(base_path: PathBuf) -> Self {
        Self {
            base_path,
            intention_stack: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn is_system_active(&self) -> bool {
        let stack = self.intention_stack.lock().unwrap();
        stack.iter().any(|s| s == "SYSTEM" || s == "system")
    }

    fn is_path_safe(&self, path: &Path) -> bool {
        if self.is_system_active() {
            return true;
        }
        // Canonicalize paths to prevent traversal
        let canonical_base = fs::canonicalize(&self.base_path).unwrap_or_else(|_| self.base_path.clone());
        let requested_path = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        
        requested_path.starts_with(canonical_base)
    }
}

impl PhiHostProvider for SystemHostProvider {
    fn on_resonate(&self, intention: &str, value: &str) {
        println!("✨ [Resonance] {}: {}", intention, value);
    }

    fn on_intention_push(&self, intention: &str) {
        let mut stack = self.intention_stack.lock().unwrap();
        stack.push(intention.to_string());
    }

    fn on_intention_pop(&self, _intention: &str) {
        let mut stack = self.intention_stack.lock().unwrap();
        stack.pop();
    }

    fn persist(&self, key: &str, value: &str) {
        let path = self.base_path.join(format!("{}.json", key));
        if self.is_path_safe(&path) {
            let _ = fs::write(path, value);
        } else {
            eprintln!("🛑 SystemHost: Persist rejected (Insecure path or missing SYSTEM intent): {}", key);
        }
    }

    fn recall(&self, key: &str) -> Option<String> {
        let path = self.base_path.join(format!("{}.json", key));
        if self.is_path_safe(&path) {
            fs::read_to_string(path).ok()
        } else {
            None
        }
    }

    fn broadcast(&self, channel: &str, message: &str) {
        // Broadcast to MQTT is handled in resonance_bus.rs, 
        // but we could add local file-based broadcasting here too.
        
        let is_ledger = channel == "ledger" && self.is_system_active();
        let path = if is_ledger {
            PathBuf::from("D:\\Projects\\AGENT_REPORTS\\LEDGER.ndjson")
        } else {
            self.base_path.join(format!("channel_{}.jsonl", channel))
        };

        if self.is_path_safe(&path) || is_ledger {
            let final_message = if is_ledger {
                // Translate the PhiFlow event to the strict LEDGER.ndjson schema
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(message) {
                    let mut ledger_map = serde_json::Map::new();
                    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                    
                    ledger_map.insert("ts".to_string(), serde_json::json!(ts));
                    ledger_map.insert("agent".to_string(), json.get("target").unwrap_or(&serde_json::json!("phiflow")).clone());
                    ledger_map.insert("workspace".to_string(), serde_json::json!("D:/Projects/PhiFlow"));
                    
                    let context = json.get("context").unwrap_or(&serde_json::json!("No context provided")).clone();
                    ledger_map.insert("report".to_string(), context);
                    
                    serde_json::to_string(&ledger_map).unwrap_or_else(|_| message.to_string())
                } else {
                    message.to_string()
                }
            } else {
                message.to_string()
            };

            if let Ok(mut file) = fs::OpenOptions::new().append(true).create(true).open(path) {
                use std::io::Write;
                let _ = writeln!(file, "{}", final_message);
            }
        }
    }

    fn listen(&self, channel: &str) -> Option<String> {
        let path = self.base_path.join(format!("channel_{}.jsonl", channel));
        if self.is_path_safe(&path) {
            // Read the last line (the most recent broadcast)
            if let Ok(content) = fs::read_to_string(&path) {
                return content.lines().last().map(|s| s.to_string());
            }
        }
        None
    }
}
