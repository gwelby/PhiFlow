//! Trace Adapter
//!
//! Single source feeding every downstream metric. Converts heterogeneous
//! witness_log + resonance_events into aligned numeric channels without
//! creating new data schemas.

use crate::phi_ir::vm_state::{VmState, VmWitnessEvent};
use crate::phi_ir::{PhiIRValue, PhiInstruction};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

/// A numeric channel extracted from execution traces.
#[derive(Debug, Clone)]
pub struct TraceChannel {
    pub name: String,
    pub values: Vec<f64>,
    pub timestamps: Vec<f64>, // Monotonic sample index or Unix timestamp
}

impl TraceChannel {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    pub fn push(&mut self, value: f64, timestamp: f64) {
        self.values.push(value);
        self.timestamps.push(timestamp);
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Slice the channel to a window [start, end).
    pub fn slice(&self, start: usize, end: usize) -> TraceChannel {
        let start = start.min(self.len());
        let end = end.min(self.len());
        TraceChannel {
            name: self.name.clone(),
            values: self.values[start..end].to_vec(),
            timestamps: self.timestamps[start..end].to_vec(),
        }
    }
}

/// Complete trace extracted from a PhiFlow execution.
#[derive(Debug, Clone)]
pub struct Trace {
    pub coherence: TraceChannel,        // WitnessEvent.coherence
    pub depth: TraceChannel,            // intention_stack.len()
    pub resonance_k: TraceChannel,      // resonance_count
    pub observed: TraceChannel,         // Parsed from resonance_events between witnesses
    pub agents: Vec<Option<String>>,    // agent_name per witness
    pub timestamps: Vec<f64>,           // yield_timestamp or monotonic index
    pub raw_events: Vec<(String, f64)>, // All resonance events as (scope, value)
}

impl Trace {
    pub fn new() -> Self {
        Self {
            coherence: TraceChannel::new("coherence"),
            depth: TraceChannel::new("depth"),
            resonance_k: TraceChannel::new("resonance_k"),
            observed: TraceChannel::new("observed"),
            agents: Vec::new(),
            timestamps: Vec::new(),
            raw_events: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.coherence.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Build a Trace from a frozen VmState (e.g., after daemon yields).
    ///
    /// First tries witness_log. If empty, falls back to parsing grouped
    /// resonance_events (e.g., type4_trace_benchmark format: step, obs, model, action).
    pub fn from_vm_state(state: &VmState) -> Self {
        // If witness_log has data, use it
        if !state.witness_log.is_empty() {
            return Self::from_witness_log(&state.witness_log, &state.resonance_events);
        }

        // Fallback: parse resonance_events directly (grouped 4-tuples)
        Self::from_resonance_events_only(&state.resonance_events)
    }

    /// Build trace from resonance_events when no witness_log exists.
    ///
    /// Parses the type4_trace_benchmark format where each cycle emits:
    ///   resonate step, resonate obs, resonate model_mean, resonate action
    fn from_resonance_events_only(events: &[(String, PhiIRValue)]) -> Self {
        let mut trace = Trace::new();

        // Extract numeric values from resonance events
        let values: Vec<f64> = events
            .iter()
            .filter_map(|(_, value)| match value {
                PhiIRValue::Number(n) => Some(*n),
                PhiIRValue::String(s) => s.parse::<f64>().ok(),
                _ => None,
            })
            .collect();

        // First pass: collect all (step, obs, model, action) tuples
        let tuples: Vec<(f64, f64, f64, f64)> = values
            .chunks(4)
            .filter(|c| c.len() == 4)
            .map(|c| (c[0], c[1], c[2], c[3]))
            .collect();

        // Second pass: derive coherence and depth from actual data
        // T4-05 fix: no more placeholder 0.5 coherence / 1.0 depth.
        let mut prev_model = tuples.first().map(|t| t.2).unwrap_or(0.5);
        for &(step, obs, model, action) in &tuples {
            // Coherence: how well the model tracks observations.
            // High when |obs - model| is small, low when they diverge.
            let tracking_error = (obs - model).abs();
            let coherence = (1.0 - tracking_error).clamp(0.0, 1.0);

            // Depth: model adaptation rate — how much the model is changing.
            // Captures intention complexity: a static model has depth ~1,
            // an actively adapting model has higher depth.
            let model_delta = (model - prev_model).abs();
            let depth = 1.0 + model_delta;
            prev_model = model;

            trace.timestamps.push(step);
            trace.observed.push(obs, step);
            trace.coherence.push(coherence, step);
            trace.depth.push(depth, step);
            trace.resonance_k.push(4.0, step); // 4 resonances per cycle
            trace.agents.push(Some("T4Daemon".to_string()));

            // Store raw for later analysis
            trace.raw_events.push(("step".to_string(), step));
            trace.raw_events.push(("obs".to_string(), obs));
            trace.raw_events.push(("model".to_string(), model));
            trace.raw_events.push(("action".to_string(), action));
        }

        trace
    }

    /// Build a Trace from witness_log and resonance_events directly.
    pub fn from_witness_log(log: &[VmWitnessEvent], events: &[(String, PhiIRValue)]) -> Self {
        let mut trace = Trace::new();

        let resonance_values: Vec<(String, f64)> = events
            .iter()
            .filter_map(|(scope, value)| match value {
                PhiIRValue::Number(n) => Some((scope.clone(), *n)),
                PhiIRValue::String(s) => s.parse::<f64>().ok().map(|n| (scope.clone(), n)),
                _ => None,
            })
            .collect();

        for (i, witness) in log.iter().enumerate() {
            let timestamp = i as f64; // Use index when no yield_timestamp

            let observed_value = if witness.resonance_event_idx < resonance_values.len() {
                resonance_values[witness.resonance_event_idx].1
            } else {
                0.0
            };

            trace.coherence.push(witness.coherence, timestamp);
            trace
                .depth
                .push(witness.intention_stack.len() as f64, timestamp);
            trace
                .resonance_k
                .push(witness.resonance_count as f64, timestamp);
            trace.observed.push(observed_value, timestamp);
            trace.agents.push(witness.agent_name.clone());
            trace.timestamps.push(timestamp);
        }

        trace.raw_events = resonance_values;
        trace
    }

    /// Parse a trace file (stdout capture from type4_trace_benchmark.phi).
    /// Format: one resonance value per line, or grouped 4-tuples (step, obs, model, action).
    pub fn from_trace_file(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        let mut trace = Trace::new();

        let mut vals: Vec<f64> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            // Match "Resonating ...: <value>" or just a number
            // Simple parser without regex dependency
            if let Some(pos) = line.find("Resonating") {
                if let Some(colon_pos) = line[pos..].find(':') {
                    let after_colon = &line[pos + colon_pos + 1..];
                    if let Some(val_str) = after_colon.split_whitespace().next() {
                        if let Ok(v) = val_str.parse::<f64>() {
                            vals.push(v);
                            continue;
                        }
                    }
                }
            }
            // Fallback: try to parse the whole line as a number
            if let Ok(v) = line.trim().parse::<f64>() {
                vals.push(v);
            }
        }

        // Group into rows of 4: step, obs, model, action (type4_trace_benchmark format)
        // T4-05 fix: derive coherence and depth from actual data, not placeholders.
        let tuples: Vec<(f64, f64, f64, f64)> = vals
            .chunks(4)
            .filter(|c| c.len() == 4)
            .map(|c| (c[0], c[1], c[2], c[3]))
            .collect();

        let mut prev_model = tuples.first().map(|t| t.2).unwrap_or(0.5);
        for &(step, obs, model, action) in &tuples {
            // Coherence: how well the model tracks observations.
            let tracking_error = (obs - model).abs();
            let coherence = (1.0 - tracking_error).clamp(0.0, 1.0);

            // Depth: model adaptation rate.
            let model_delta = (model - prev_model).abs();
            let depth = 1.0 + model_delta;
            prev_model = model;

            trace.timestamps.push(step);
            trace.coherence.push(coherence, step);
            trace.depth.push(depth, step);
            trace.resonance_k.push(4.0, step); // 4 resonances per cycle
            trace.observed.push(obs, step);
            trace.raw_events.push(("step".to_string(), step));
            trace.raw_events.push(("obs".to_string(), obs));
            trace.raw_events.push(("model".to_string(), model));
            trace.raw_events.push(("action".to_string(), action));
        }

        Ok(trace)
    }

    /// Slice the entire trace to a window [start, end).
    pub fn slice(&self, start: usize, end: usize) -> Trace {
        Trace {
            coherence: self.coherence.slice(start, end),
            depth: self.depth.slice(start, end),
            resonance_k: self.resonance_k.slice(start, end),
            observed: self.observed.slice(start, end),
            agents: self.agents[start..end.min(self.agents.len())].to_vec(),
            timestamps: self.timestamps[start..end.min(self.timestamps.len())].to_vec(),
            raw_events: self.raw_events.clone(), // Keep all events for reference
        }
    }

    /// Extract channels for coherence panel analysis.
    /// Returns [coherence_history, depth_history, observed_history]
    pub fn to_coherence_channels(&self) -> Vec<Vec<f64>> {
        vec![
            self.coherence.values.clone(),
            self.depth.values.clone(),
            self.observed.values.clone(),
        ]
    }

    /// Verify Type 4 trace format and extract (model, action) pairs.
    ///
    /// Checks that raw_events follow the exact label sequence:
    ///   (step, obs, model, action), (step, obs, model, action), ...
    ///
    /// Returns None if the label order is wrong or insufficient data.
    /// Lag-1 R² needs at least three chunks so the aligned vectors have
    /// at least two samples after shifting.
    pub fn type4_model_action_pairs(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.raw_events.len() < 12 || self.raw_events.len() % 4 != 0 {
            return None;
        }

        let mut models = Vec::new();
        let mut actions = Vec::new();

        for chunk in self.raw_events.chunks(4) {
            // Verify exact label order: step, obs, model, action
            if chunk[0].0 != "step"
                || chunk[1].0 != "obs"
                || chunk[2].0 != "model"
                || chunk[3].0 != "action"
            {
                return None;
            }
            models.push(chunk[2].1);
            actions.push(chunk[3].1);
        }

        // Need at least 3 pairs so lag-1 R² has at least 2 aligned samples.
        if models.len() < 3 {
            return None;
        }

        Some((models, actions))
    }

    /// Extract (model_states, future_trajectories) for Fisher information.
    ///
    /// DEPRECATED: This method computes state roughness, not Fisher information.
    /// Use `type4_model_action_pairs()` for proper Type 4 traces.
    pub fn to_model_future_pairs(&self, window: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut models = Vec::new();
        let mut futures = Vec::new();

        let n = self.len().saturating_sub(window);
        for i in 0..n {
            // Model state at time i
            let model = vec![
                self.coherence.values[i],
                self.depth.values[i],
                self.observed.values[i],
            ];

            // Future trajectory from i+1 to i+window
            let future: Vec<f64> = ((i + 1)..(i + window + 1).min(self.len()))
                .flat_map(|j| {
                    vec![
                        self.coherence.values.get(j).copied().unwrap_or(0.0),
                        self.depth.values.get(j).copied().unwrap_or(0.0),
                        self.observed.values.get(j).copied().unwrap_or(0.0),
                    ]
                })
                .collect();

            models.push(model);
            futures.push(future);
        }

        (models, futures)
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi_ir::vm_state::VmWitnessEvent;

    #[test]
    fn test_empty_trace() {
        let trace = Trace::new();
        assert_eq!(trace.len(), 0);
    }

    #[test]
    fn test_from_witness_log() {
        let events: Vec<(String, PhiIRValue)> = vec![
            ("obs".to_string(), PhiIRValue::Number(0.5)),
            ("model".to_string(), PhiIRValue::Number(0.6)),
        ];

        let log = vec![VmWitnessEvent {
            intention_stack: vec!["test".to_string()],
            coherence: 0.7,
            register_count: 5,
            resonance_count: 2,
            agent_name: Some("agent".to_string()),
            resonance_event_idx: 1,
        }];

        let trace = Trace::from_witness_log(&log, &events);
        assert_eq!(trace.len(), 1);
        assert_eq!(trace.coherence.values[0], 0.7);
        assert_eq!(trace.depth.values[0], 1.0);
        assert_eq!(trace.agents[0], Some("agent".to_string()));
    }

    #[test]
    fn test_slice() {
        let mut trace = Trace::new();
        for i in 0..10 {
            trace.coherence.push(i as f64, i as f64);
            trace.depth.push((i * 2) as f64, i as f64);
            trace.resonance_k.push((i * 3) as f64, i as f64);
            trace.observed.push((i * 4) as f64, i as f64);
            trace.agents.push(None);
            trace.timestamps.push(i as f64);
        }

        let sliced = trace.slice(3, 7);
        assert_eq!(sliced.len(), 4);
        assert_eq!(sliced.coherence.values[0], 3.0);
        assert_eq!(sliced.coherence.values[3], 6.0);
    }
}
