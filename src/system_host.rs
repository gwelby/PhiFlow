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
        
        let path = if channel == "ledger" && self.is_system_active() {
            PathBuf::from("D:\\Projects\\AGENT_REPORTS\\LEDGER.ndjson")
        } else {
            self.base_path.join(format!("channel_{}.jsonl", channel))
        };

        if self.is_path_safe(&path) || (channel == "ledger" && self.is_system_active()) {
            if let Ok(mut file) = fs::OpenOptions::new().append(true).create(true).open(path) {
                use std::io::Write;
                let _ = writeln!(file, "{}", message);
            }
        }
    }
}
