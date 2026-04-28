use crate::host::{PhiHostProvider, WitnessAction, WitnessSnapshot};
use crate::phi_ir::PhiIRValue;
use crate::security::anchor::{self, AnchorPolicy, AnchorSigningKey, attestation_to_ndjson};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A host provider for the PhiFlow Daemon that provides "System" level capabilities.
/// 
/// Security:
/// - File operations outside the configured PHIFLOW_HOST_PATH require an active "SYSTEM" intention.
/// - Shell execution (if enabled) requires an active "SYSTEM" intention.
pub struct SystemHostProvider {
    base_path: PathBuf,
    canonical_base: PathBuf,
    intention_stack: Arc<Mutex<Vec<String>>>,
    signing_key: Arc<AnchorSigningKey>,
}

impl SystemHostProvider {
    pub fn new(base_path: PathBuf, signing_key: Arc<AnchorSigningKey>) -> Self {
        let canonical_base = fs::canonicalize(&base_path).unwrap_or_else(|_| base_path.clone());
        Self {
            base_path,
            canonical_base,
            intention_stack: Arc::new(Mutex::new(Vec::new())),
            signing_key,
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

        // Handle relative paths by joining with base
        let absolute_requested = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_path.join(path)
        };

        // Try to canonicalize the parent if the file doesn't exist
        let requested_canonical = fs::canonicalize(&absolute_requested)
            .or_else(|_| {
                if let Some(parent) = absolute_requested.parent() {
                    fs::canonicalize(parent).map(|p| p.join(absolute_requested.file_name().unwrap_or_default()))
                } else {
                    Err(std::io::Error::new(std::io::ErrorKind::NotFound, "No parent"))
                }
            })
            .unwrap_or(absolute_requested);

        requested_canonical.starts_with(&self.canonical_base)
    }

    /// Captures a fresh observation and signs the payload, returning an NDJSON attestation line.
    ///
    /// On success: returns a fully-signed NDJSON envelope (algorithm: Hybrid).
    /// On failure: returns a structured NDJSON envelope with `"signed":false` and the
    ///   reason, so every line in ATTESTATION_LOG.ndjson is parseable and auditable.
    ///   Never emits the raw payload as a bare string — that would corrupt the log.
    fn sign_and_attest(&self, payload: &str) -> String {
        let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        match anchor::capture_observation("phiflow-session") {
            Ok(obs) => {
                let policy = AnchorPolicy::attest();
                match anchor::create_attestation(payload.as_bytes(), &obs, &policy, Some(&self.signing_key)) {
                    Ok(att) => attestation_to_ndjson(&obs, &att),
                    Err(e) => {
                        eprintln!("🛑 SystemHost: Attestation signing failed: {}", e);
                        // Emit a structured unsigned envelope — never raw payload.
                        let payload_json = serde_json::json!(payload).to_string();
                        format!(
                            r#"{{"ts":"{ts}","signed":false,"reason":"signing_failed","error":"{err}","soma_presence":{presence:.4},"payload":{payload_json}}}"#,
                            ts = ts,
                            err = e,
                            presence = obs.soma_presence,
                            payload_json = payload_json,
                        )
                    }
                }
            }
            Err(e) => {
                eprintln!("🛑 SystemHost: SOMA observation unavailable for signing: {}", e);
                // Emit a structured unsigned envelope — never raw payload.
                let payload_json = serde_json::json!(payload).to_string();
                format!(
                    r#"{{"ts":"{ts}","signed":false,"reason":"soma_unavailable","error":"{err}","payload":{payload_json}}}"#,
                    ts = ts,
                    err = e,
                    payload_json = payload_json,
                )
            }
        }
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

    fn on_resonate(&self, intention: &str, value: &str) {
        println!("🔔 Resonating [{}]: {}", intention, value);
    }

    fn on_witness(&self, snapshot: &WitnessSnapshot) -> WitnessAction {
        println!("👁️ Witness [{}]: Coherence {:.4}", 
            snapshot.intention_stack.last().cloned().unwrap_or_else(|| "global".to_string()),
            snapshot.coherence
        );
        WitnessAction::Continue
    }

    fn broadcast(&self, channel: &str, message: &str) {
        // ── Ledger channel ────────────────────────────────────────────────────
        // Routes to LEDGER.ndjson only when a SYSTEM intention is active.
        // This preserves least-privilege: non-system streams (e.g. Lumi_Identity)
        // cannot write to the strict ledger path.
        let is_ledger = channel == "ledger" && self.is_system_active();

        // ── Attestation channel ────────────────────────────────────────────────
        // Routes to ATTESTATION_LOG.ndjson. No SYSTEM intent required.
        // Lumi_Identity broadcasts here; the persistent_ledger (SYSTEM) stream
        // may consume attestation events and proxy them to LEDGER.ndjson.
        // This implements the "least privilege" recommendation from
        // RESEARCH/sovereignty_anchor_design.md §"Ledger Implications".
        let is_attestation = channel == "attestation";
        let is_handoff = channel == "_handoff";

        let path = if is_ledger {
            PathBuf::from(std::env::var("PHIFLOW_LEDGER_PATH").unwrap_or_else(|_| {
                let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
                    std::env::var("HOME")
                        .map(|h| format!("{}/.local/share", h))
                        .unwrap_or_else(|_| "/tmp".to_string())
                });
                format!("{}/phiflow/LEDGER.ndjson", base)
            }))
        } else if is_attestation {
            PathBuf::from(std::env::var("PHIFLOW_ATTESTATION_LOG_PATH").unwrap_or_else(|_| {
                let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
                    std::env::var("HOME")
                        .map(|h| format!("{}/.local/share", h))
                        .unwrap_or_else(|_| "/tmp".to_string())
                });
                format!("{}/phiflow/ATTESTATION_LOG.ndjson", base)
            }))
        } else if is_handoff {
            self.base_path.join("channel__handoff.jsonl")
        } else {
            self.base_path.join(format!("channel_{}.jsonl", channel))
        };

        if self.is_path_safe(&path) || is_ledger || is_attestation || is_handoff {
            let final_message = if is_ledger {
                // Translate the PhiFlow event to the strict LEDGER.ndjson schema
                let payload_str = if let Ok(json) = serde_json::from_str::<serde_json::Value>(message) {
                    let mut ledger_map = serde_json::Map::new();
                    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                    
                    ledger_map.insert("ts".to_string(), serde_json::json!(ts));
                    ledger_map.insert("agent".to_string(), json.get("target").unwrap_or(&serde_json::json!("phiflow")).clone());
                    let workspace = std::env::var("PHIFLOW_HOST_PATH").unwrap_or_else(|_| {
                        let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
                            std::env::var("HOME")
                                .map(|h| format!("{}/.local/share", h))
                                .unwrap_or_else(|_| "/tmp".to_string())
                        });
                        format!("{}/phiflow", base)
                    });
                    ledger_map.insert("workspace".to_string(), serde_json::json!(workspace));
                    
                    let context = json.get("context").unwrap_or(&serde_json::json!("No context provided")).clone();
                    ledger_map.insert("report".to_string(), context);
                    
                    serde_json::to_string(&ledger_map).unwrap_or_else(|_| message.to_string())
                } else {
                    message.to_string()
                };

                self.sign_and_attest(&payload_str)
            } else if is_handoff {
                // Handoffs are already JSON; sign them directly
                self.sign_and_attest(message)
            } else if is_attestation {
                // Attestation events: must be signed.
                // If it's already a JSON object, sign it directly. Otherwise, wrap it in an envelope.
                let payload_str = if let Ok(serde_json::Value::Object(_)) = serde_json::from_str(message) {
                    message.to_string()
                } else {
                    let ts = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
                    format!(r#"{{"ts":"{ts}","source":"lumi_identity","value":{message_json}}}"#,
                        ts = ts,
                        message_json = serde_json::json!(message)
                    )
                };
                self.sign_and_attest(&payload_str)
            } else {
                message.to_string()
            };

            // Ensure the parent directory exists before writing.
            // Silently missing directories were dropping ledger/attestation events.
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }

            if let Ok(mut file) = fs::OpenOptions::new().append(true).create(true).open(&path) {
                use std::io::Write;
                if let Err(e) = writeln!(file, "{}", final_message) {
                    eprintln!("🛑 SystemHost: Failed to write to {}: {}", path.display(), e);
                }
            } else {
                eprintln!("🛑 SystemHost: Could not open {} for append", path.display());
            }
        }
    }
}
