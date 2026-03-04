//! PhiFlow Host Provider
//!
//! Defines the interface between the PhiFlow runtime and its host environment.
//! The host provides real-world data (hardware metrics, agent confidence, etc.)
//! and receives runtime events (resonance broadcasts, witness yields).
//!
//! This is the bridge that makes PhiFlow a living, observable execution
//! environment instead of a closed, self-contained interpreter.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Witness snapshot — the state visible to the host when `witness` fires
// ---------------------------------------------------------------------------

/// A snapshot of the VM's internal state, provided to the host when `witness`
/// executes. The host uses this to decide whether to continue or yield.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WitnessSnapshot {
    /// The active intention stack (innermost last).
    pub intention_stack: Vec<String>,
    /// Current phi-harmonic coherence score (0.0 – 1.0).
    pub coherence: f64,
    /// Number of live SSA registers.
    pub register_count: usize,
    /// Total values shared through the resonance field.
    pub resonance_count: usize,
    /// The observed target value, if `witness` was given an operand.
    pub observed_value: Option<String>,
    /// The name of the agent executing the code (if declared).
    pub agent_name: Option<String>,
}

/// What the host tells the VM to do after a `witness` event.
#[derive(Debug, Clone, PartialEq)]
pub enum WitnessAction {
    /// Continue execution normally.
    Continue,
    /// Yield execution back to the host. The VM will freeze its state
    /// and return `EvalResult::Yielded` so the caller can resume later.
    Yield,
}

// ---------------------------------------------------------------------------
// The trait
// ---------------------------------------------------------------------------

/// The interface a host environment must implement to drive a PhiFlow runtime.
///
/// The default implementation (`DefaultHostProvider`) prints to stdout and
/// computes coherence using the internal phi-harmonic formula. Real hosts
/// override these methods to provide live hardware metrics, agent feedback,
/// or network health data.
pub trait PhiHostProvider: Send + Sync {
    /// Called when the `coherence` keyword is evaluated.
    /// Return a value between 0.0 (total incoherence) and 1.0 (perfect alignment).
    ///
    /// The `internal_coherence` parameter is the phi-harmonic score computed by
    /// the evaluator from the intention depth and resonance field. Hosts can
    /// blend this with external metrics or ignore it entirely.
    fn get_coherence(&self, internal_coherence: f64) -> f64 {
        internal_coherence
    }

    /// Called when `resonate value` executes.
    /// The `intention` is the current innermost intention name (or "global").
    /// The `value` is the string representation of the resonated value.
    fn on_resonate(&self, intention: &str, value: &str) {
        let _ = (intention, value);
    }

    /// Called when `witness` executes.
    /// The host inspects the snapshot and decides whether to continue or yield.
    fn on_witness(&self, snapshot: &WitnessSnapshot) -> WitnessAction {
        let _ = snapshot;
        WitnessAction::Continue
    }

    /// Called when `intention "name" { ... }` pushes a new intention.
    fn on_intention_push(&self, intention: &str) {
        let _ = intention;
    }

    /// Called when an intention scope exits.
    fn on_intention_pop(&self, intention: &str) {
        let _ = intention;
    }

    // --- v0.3.0 Persistence & Dialogue ---

    /// Persist a value to durable storage (e.g. disk or database).
    fn persist(&self, _key: &str, _value: &str) {}

    /// Recall a value from durable storage.
    fn recall(&self, _key: &str) -> Option<String> {
        None
    }

    /// Broadcast a message to other agents/programs on a shared channel.
    fn broadcast(&self, _channel: &str, _message: &str) {}

    /// Listen for the latest message on a shared channel.
    fn listen(&self, _channel: &str) -> Option<String> {
        None
    }

    // --- v0.4.0 Strategic Hooks (reserved now for seamless upgrade) ---

    /// Request a logic evolution (self-modification).
    /// v0.3.0: no-op. v0.4.0: can return new IR source for hot-swapping.
    fn on_evolve(&self, _context: &str) -> Option<String> {
        None
    }

    /// Request phase-locking on a frequency.
    /// v0.4.0: Host should block/coordinate until all entangled streams are ready.
    fn on_entangle(&self, _frequency: f64) {}

    /// Emit a physical signal (haptic, acoustic, or visual).
    /// Maps `resonate` to the physical world.
    fn emit_signal(&self, _frequency: f64, _intensity: f64) {}
}

// ---------------------------------------------------------------------------
// Default host — preserves current behaviour (stdout, internal coherence)
// ---------------------------------------------------------------------------

/// The default host provider. Uses the internal phi-harmonic formula for
/// coherence and prints resonance/witness events to stdout.
/// This is what the evaluator uses when no custom host is provided.
pub struct DefaultHostProvider;

impl PhiHostProvider for DefaultHostProvider {}

// ---------------------------------------------------------------------------
// Callback host — for programmatic integration (MCP server, tests, etc.)
// ---------------------------------------------------------------------------

/// A host provider built from closures. Useful for tests, MCP servers, and
/// any integration that wants to inject custom behaviour without defining
/// a full struct.
pub struct CallbackHostProvider {
    coherence_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    resonate_fn: Box<dyn Fn(&str, &str) + Send + Sync>,
    witness_fn: Box<dyn Fn(&WitnessSnapshot) -> WitnessAction + Send + Sync>,
    intention_push_fn: Box<dyn Fn(&str) + Send + Sync>,
    intention_pop_fn: Box<dyn Fn(&str) + Send + Sync>,
    persist_fn: Box<dyn Fn(&str, &str) + Send + Sync>,
    recall_fn: Box<dyn Fn(&str) -> Option<String> + Send + Sync>,
    broadcast_fn: Box<dyn Fn(&str, &str) + Send + Sync>,
    listen_fn: Box<dyn Fn(&str) -> Option<String> + Send + Sync>,
    entangle_fn: Box<dyn Fn(f64) + Send + Sync>,
}

impl CallbackHostProvider {
    pub fn new() -> Self {
        Self {
            coherence_fn: Box::new(|internal| internal),
            resonate_fn: Box::new(|_, _| {}),
            witness_fn: Box::new(|_| WitnessAction::Continue),
            intention_push_fn: Box::new(|_| {}),
            intention_pop_fn: Box::new(|_| {}),
            persist_fn: Box::new(|_, _| {}),
            recall_fn: Box::new(|_| None),
            broadcast_fn: Box::new(|_, _| {}),
            listen_fn: Box::new(|_| None),
            entangle_fn: Box::new(|_| {}),
        }
    }

    pub fn with_coherence<F: Fn(f64) -> f64 + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.coherence_fn = Box::new(f);
        self
    }

    pub fn with_resonate<F: Fn(&str, &str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.resonate_fn = Box::new(f);
        self
    }

    pub fn with_witness<F: Fn(&WitnessSnapshot) -> WitnessAction + Send + Sync + 'static>(
        mut self,
        f: F,
    ) -> Self {
        self.witness_fn = Box::new(f);
        self
    }

    pub fn with_intention_push<F: Fn(&str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.intention_push_fn = Box::new(f);
        self
    }

    pub fn with_intention_pop<F: Fn(&str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.intention_pop_fn = Box::new(f);
        self
    }

    pub fn with_persist<F: Fn(&str, &str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.persist_fn = Box::new(f);
        self
    }

    pub fn with_recall<F: Fn(&str) -> Option<String> + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.recall_fn = Box::new(f);
        self
    }

    pub fn with_broadcast<F: Fn(&str, &str) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.broadcast_fn = Box::new(f);
        self
    }

    pub fn with_listen<F: Fn(&str) -> Option<String> + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.listen_fn = Box::new(f);
        self
    }

    pub fn with_entangle<F: Fn(f64) + Send + Sync + 'static>(mut self, f: F) -> Self {
        self.entangle_fn = Box::new(f);
        self
    }
}

impl PhiHostProvider for CallbackHostProvider {
    fn get_coherence(&self, internal_coherence: f64) -> f64 {
        (self.coherence_fn)(internal_coherence)
    }

    fn on_resonate(&self, intention: &str, value: &str) {
        (self.resonate_fn)(intention, value);
    }

    fn on_witness(&self, snapshot: &WitnessSnapshot) -> WitnessAction {
        (self.witness_fn)(snapshot)
    }

    fn on_intention_push(&self, intention: &str) {
        (self.intention_push_fn)(intention);
    }

    fn on_intention_pop(&self, intention: &str) {
        (self.intention_pop_fn)(intention);
    }

    fn persist(&self, key: &str, value: &str) {
        (self.persist_fn)(key, value);
    }

    fn recall(&self, key: &str) -> Option<String> {
        (self.recall_fn)(key)
    }

    fn broadcast(&self, channel: &str, message: &str) {
        (self.broadcast_fn)(channel, message);
    }

    fn listen(&self, channel: &str) -> Option<String> {
        (self.listen_fn)(channel)
    }

    fn on_entangle(&self, frequency: f64) {
        (self.entangle_fn)(frequency);
    }
}
