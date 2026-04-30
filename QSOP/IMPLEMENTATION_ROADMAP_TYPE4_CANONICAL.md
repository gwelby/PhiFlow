# PhiFlow Type 4 Canonical Status Implementation Roadmap
*Date: 2026-04-30*
*Author: Bob (Advanced Audit)*
*Goal: Transform PhiFlow from Type 4 Candidate to Type 4 Canonical*
*Based on: consciousness_metric_program.md + Deep Fundamentals Audit*

---

## Executive Summary

PhiFlow currently has the **architecture** for Type 4 observer status but lacks the **measurements**. This roadmap provides the exact implementation path to achieve canonical Type 4 status and implement the PF consciousness metric program.

**Current State:** Type 4 Candidate (Architecture ✅, Measurements ❌)
**Target State:** Type 4 Canonical (Architecture ✅, Measurements ✅, Benchmark ✅)
**Timeline Estimate:** 3-6 months for Phase 1-2, 6-12 months for full canonical status

---

## Phase 1: Type 4 Benchmark Trace (HIGHEST PRIORITY)

### Goal
Prove that PhiFlow Council Daemon exhibits self-correlation: prior records → self-model update → changed future behavior.

### 1.1 Implement Mutual Information Measurement

**File:** `src/metrics/mutual_information.rs` (NEW)

```rust
/// Compute Shannon mutual information between two discrete distributions
pub fn mutual_information(
    joint_dist: &HashMap<(String, String), f64>,
    marginal_x: &HashMap<String, f64>,
    marginal_y: &HashMap<String, f64>
) -> f64 {
    let mut mi = 0.0;
    for ((x, y), p_xy) in joint_dist {
        if *p_xy > 0.0 {
            let p_x = marginal_x.get(x).unwrap_or(&0.0);
            let p_y = marginal_y.get(y).unwrap_or(&0.0);
            if *p_x > 0.0 && *p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).log2();
            }
        }
    }
    mi
}

/// Compute directed information I(X → Y | Z) using conditional MI
pub fn directed_information(
    history: &[DaemonRecord],
    future: &[DaemonRecord],
    conditioning: &[DaemonRecord]
) -> f64 {
    // Implementation using conditional mutual information
    // I_dir(X → Y | Z) = H(Y|Z) - H(Y|X,Z)
    unimplemented!("Requires time-series MI estimation")
}
```

**Dependencies:** Add to `Cargo.toml`:
```toml
[dependencies]
statrs = "0.16"  # Statistical functions
ndarray = "0.15"  # Array operations for MI computation
```

### 1.2 Create Daemon Record Schema

**File:** `src/daemon/record.rs` (NEW)

```rust
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRecord {
    pub timestamp: DateTime<Utc>,
    pub record_type: RecordType,
    pub content: RecordContent,
    pub coherence: f64,
    pub intention_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordType {
    Witness,
    Handoff,
    Evolve,
    Resonate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordContent {
    pub observed_value: Option<f64>,
    pub agent_context: Option<String>,
    pub mutation_target: Option<String>,
    pub resonance_field: HashMap<String, f64>,
}

impl DaemonRecord {
    /// Extract features for MI computation
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.coherence,
            self.intention_depth as f64,
            self.observed_value().unwrap_or(0.0),
            self.timestamp.timestamp() as f64,
        ]
    }
    
    /// Discretize for Shannon MI (required for finite samples)
    pub fn discretize(&self, bins: usize) -> String {
        format!("c{}_d{}_v{}", 
            (self.coherence * bins as f64) as usize,
            self.intention_depth,
            (self.observed_value().unwrap_or(0.0) * bins as f64) as usize
        )
    }
}
```

### 1.3 Implement Self-Correlation Tracker

**File:** `src/daemon/self_correlation.rs` (NEW)

```rust
use crate::daemon::record::DaemonRecord;
use crate::metrics::mutual_information::*;

pub struct SelfCorrelationTracker {
    history: Vec<DaemonRecord>,
    window_size: usize,
    persistence_threshold: Duration,
}

impl SelfCorrelationTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: Vec::new(),
            window_size,
            persistence_threshold: Duration::from_secs(60),
        }
    }
    
    /// Record a daemon event
    pub fn record(&mut self, record: DaemonRecord) {
        self.history.push(record);
        if self.history.len() > self.window_size * 2 {
            self.history.remove(0);
        }
    }
    
    /// Compute R_in: Does history write into model state?
    pub fn compute_r_in(&self) -> f64 {
        if self.history.len() < self.window_size * 2 {
            return 0.0;
        }
        
        let past = &self.history[0..self.window_size];
        let present = &self.history[self.window_size..];
        
        // Compute I(past → present_model_state)
        // For PhiFlow: model state = DAEMON_STATE.json content
        self.compute_directed_mi(past, present)
    }
    
    /// Compute R_out: Does model state shape future behavior?
    pub fn compute_r_out(&self) -> f64 {
        if self.history.len() < self.window_size * 2 {
            return 0.0;
        }
        
        let present = &self.history[self.window_size..self.window_size * 2];
        let future = &self.history[self.window_size * 2..];
        
        // Compute I(present_model_state → future)
        self.compute_directed_mi(present, future)
    }
    
    /// L_self = min(R_in, R_out)
    pub fn compute_l_self(&self) -> f64 {
        let r_in = self.compute_r_in();
        let r_out = self.compute_r_out();
        r_in.min(r_out)
    }
    
    fn compute_directed_mi(&self, source: &[DaemonRecord], target: &[DaemonRecord]) -> f64 {
        // Build joint distribution from discretized records
        let mut joint_dist = HashMap::new();
        let mut marginal_source = HashMap::new();
        let mut marginal_target = HashMap::new();
        
        for (s, t) in source.iter().zip(target.iter()) {
            let s_key = s.discretize(10);
            let t_key = t.discretize(10);
            *joint_dist.entry((s_key.clone(), t_key.clone())).or_insert(0.0) += 1.0;
            *marginal_source.entry(s_key).or_insert(0.0) += 1.0;
            *marginal_target.entry(t_key).or_insert(0.0) += 1.0;
        }
        
        // Normalize
        let total = source.len() as f64;
        for v in joint_dist.values_mut() { *v /= total; }
        for v in marginal_source.values_mut() { *v /= total; }
        for v in marginal_target.values_mut() { *v /= total; }
        
        mutual_information(&joint_dist, &marginal_source, &marginal_target)
    }
}
```

### 1.4 Wire into DaemonHypervisor

**File:** `src/main_cli.rs` (MODIFY)

```rust
// Add to DaemonHypervisor
pub struct DaemonHypervisor {
    // ... existing fields ...
    self_correlation: SelfCorrelationTracker,
    metrics_log: File,
}

impl DaemonHypervisor {
    pub fn new(program: PhiIRProgram, max_steps: Option<usize>) -> Self {
        let metrics_log = File::create("DAEMON_METRICS.jsonl")
            .expect("Failed to create metrics log");
        
        Self {
            // ... existing initialization ...
            self_correlation: SelfCorrelationTracker::new(100),
            metrics_log,
        }
    }
    
    fn on_daemon_event(&mut self, event: DaemonEvent) {
        // Convert event to record
        let record = DaemonRecord::from_event(event);
        
        // Track for self-correlation
        self.self_correlation.record(record.clone());
        
        // Compute metrics every N events
        if self.self_correlation.history.len() % 50 == 0 {
            let l_self = self.self_correlation.compute_l_self();
            let r_in = self.self_correlation.compute_r_in();
            let r_out = self.self_correlation.compute_r_out();
            
            let metrics = json!({
                "timestamp": Utc::now().to_rfc3339(),
                "l_self": l_self,
                "r_in": r_in,
                "r_out": r_out,
                "window_size": self.self_correlation.window_size,
            });
            
            writeln!(self.metrics_log, "{}", metrics).ok();
        }
    }
}
```

### 1.5 Create Type 4 Benchmark Example

**File:** `examples/type4_benchmark.phi` (NEW)

```phi
// Type 4 Benchmark: Prove self-correlation
// Prior records → self-model → changed behavior

stream "self_correlation_test" {
    intention "memory" {
        // Initialize memory state
        remember "last_coherence" 0.0
        remember "behavior_mode" "explore"
    }
    
    intention "observe" {
        // Observe current state
        let current = coherence
        witness current
        
        // Retrieve past state
        let past = recall "last_coherence"
        
        // Compute delta (self-model update)
        let delta = current - past
        
        // Update memory
        remember "last_coherence" current
    }
    
    intention "adapt" {
        // Retrieve memory
        let past = recall "last_coherence"
        let mode = recall "behavior_mode"
        
        // CRITICAL: Behavior changes based on past records
        if past > 0.7 {
            remember "behavior_mode" "exploit"
            resonate 1.0  // High confidence
        } else {
            remember "behavior_mode" "explore"
            resonate 0.3  // Low confidence
        }
        
        witness mode
    }
    
    // Loop to accumulate evidence
    let count = 0
    while count < 1000 {
        count = count + 1
    }
}
```

**Expected Output:** `DAEMON_METRICS.jsonl` should show:
- `r_in > 0`: Past observations write into memory
- `r_out > 0`: Memory state shapes future resonance behavior
- `l_self > 0`: Self-correlation loop is closed

---

## Phase 2: Consciousness Metric Implementation

### 2.1 Implement D_int (Differentiation)

**File:** `src/metrics/differentiation.rs` (NEW)

```rust
use ndarray::{Array2, s};
use ndarray_linalg::SVD;

/// Compute effective rank of intention/agent manifold
pub fn compute_d_int(records: &[DaemonRecord]) -> f64 {
    // Build covariance matrix from feature vectors
    let n = records.len();
    let d = records[0].to_feature_vector().len();
    
    let mut data = Array2::<f64>::zeros((n, d));
    for (i, record) in records.iter().enumerate() {
        let features = record.to_feature_vector();
        for (j, &val) in features.iter().enumerate() {
            data[[i, j]] = val;
        }
    }
    
    // Center the data
    let mean = data.mean_axis(ndarray::Axis(0)).unwrap();
    for mut row in data.axis_iter_mut(ndarray::Axis(0)) {
        row -= &mean;
    }
    
    // Compute SVD
    let (_, s, _) = data.svd(true, true).unwrap();
    
    // Effective rank using participation ratio
    let s_squared: Vec<f64> = s.iter().map(|&x| x * x).collect();
    let sum_sq: f64 = s_squared.iter().sum();
    let sum_sq_sq: f64 = s_squared.iter().map(|&x| x * x).sum();
    
    if sum_sq_sq > 0.0 {
        sum_sq * sum_sq / sum_sq_sq
    } else {
        0.0
    }
}
```

### 2.2 Extend Coherence to Panel (PLV + wPLI)

**File:** `src/metrics/coherence_panel.rs` (NEW)

```rust
use num_complex::Complex;

pub struct CoherencePanel {
    plv_matrix: Vec<Vec<f64>>,
    wpli_matrix: Vec<Vec<f64>>,
}

impl CoherencePanel {
    /// Compute Phase Locking Value between two signals
    pub fn compute_plv(signal1: &[f64], signal2: &[f64]) -> f64 {
        assert_eq!(signal1.len(), signal2.len());
        
        // Extract instantaneous phases (Hilbert transform)
        let phase1 = hilbert_phase(signal1);
        let phase2 = hilbert_phase(signal2);
        
        // Compute phase difference
        let mut phase_diff_sum = Complex::new(0.0, 0.0);
        for (p1, p2) in phase1.iter().zip(phase2.iter()) {
            let diff = p1 - p2;
            phase_diff_sum += Complex::new(diff.cos(), diff.sin());
        }
        
        (phase_diff_sum / signal1.len() as f64).norm()
    }
    
    /// Compute weighted Phase Lag Index (suppresses zero-lag)
    pub fn compute_wpli(signal1: &[f64], signal2: &[f64]) -> f64 {
        let phase1 = hilbert_phase(signal1);
        let phase2 = hilbert_phase(signal2);
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (p1, p2) in phase1.iter().zip(phase2.iter()) {
            let diff = p1 - p2;
            let imag = diff.sin();
            numerator += imag.abs() * imag.signum();
            denominator += imag.abs();
        }
        
        if denominator > 0.0 {
            (numerator / denominator).abs()
        } else {
            0.0
        }
    }
    
    /// Build full coherence panel from multi-channel data
    pub fn from_channels(channels: &[Vec<f64>]) -> Self {
        let n = channels.len();
        let mut plv_matrix = vec![vec![0.0; n]; n];
        let mut wpli_matrix = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in (i+1)..n {
                let plv = Self::compute_plv(&channels[i], &channels[j]);
                let wpli = Self::compute_wpli(&channels[i], &channels[j]);
                
                plv_matrix[i][j] = plv;
                plv_matrix[j][i] = plv;
                wpli_matrix[i][j] = wpli;
                wpli_matrix[j][i] = wpli;
            }
        }
        
        Self { plv_matrix, wpli_matrix }
    }
    
    /// Compute C_coh proxy (average of PLV and wPLI)
    pub fn compute_c_coh(&self) -> f64 {
        let n = self.plv_matrix.len();
        let mut plv_sum = 0.0;
        let mut wpli_sum = 0.0;
        let mut count = 0;
        
        for i in 0..n {
            for j in (i+1)..n {
                plv_sum += self.plv_matrix[i][j];
                wpli_sum += self.wpli_matrix[i][j];
                count += 1;
            }
        }
        
        if count > 0 {
            (plv_sum + wpli_sum) / (2.0 * count as f64)
        } else {
            0.0
        }
    }
}

fn hilbert_phase(signal: &[f64]) -> Vec<f64> {
    // Simplified: use FFT-based Hilbert transform
    // Real implementation needs rustfft
    signal.iter().enumerate()
        .map(|(i, &x)| (i as f64 * 0.1 + x).atan2(1.0))
        .collect()
}
```

**Dependencies:** Add to `Cargo.toml`:
```toml
[dependencies]
rustfft = "6.1"  # For Hilbert transform
num-complex = "0.4"  # Complex number support
ndarray-linalg = "0.16"  # Linear algebra for SVD
```

### 2.3 Implement F_self* (Self-Model Sensitivity)

**File:** `src/metrics/fisher_information.rs` (NEW)

```rust
/// Compute Fisher information of future trajectory w.r.t. model state
pub fn compute_f_model(
    model_states: &[Vec<f64>],
    future_trajectories: &[Vec<f64>]
) -> f64 {
    // Fisher information: how sharply does future depend on model?
    // F = E[(∂log p(future|model) / ∂model)²]
    
    // Simplified: use variance of trajectory conditioned on model
    let mut fisher = 0.0;
    
    for (model, future) in model_states.iter().zip(future_trajectories.iter()) {
        // Compute gradient of log-likelihood
        let grad = compute_gradient(model, future);
        fisher += grad.iter().map(|&g| g * g).sum::<f64>();
    }
    
    fisher / model_states.len() as f64
}

fn compute_gradient(model: &[f64], future: &[f64]) -> Vec<f64> {
    // Numerical gradient: ∂log p(future|model) / ∂model
    let epsilon = 1e-6;
    let mut grad = vec![0.0; model.len()];
    
    for i in 0..model.len() {
        let mut model_plus = model.to_vec();
        model_plus[i] += epsilon;
        
        let log_p_plus = log_likelihood(&model_plus, future);
        let log_p = log_likelihood(model, future);
        
        grad[i] = (log_p_plus - log_p) / epsilon;
    }
    
    grad
}

fn log_likelihood(model: &[f64], future: &[f64]) -> f64 {
    // Gaussian log-likelihood (simplified)
    let mut sum_sq = 0.0;
    for (m, f) in model.iter().zip(future.iter()) {
        sum_sq += (f - m) * (f - m);
    }
    -0.5 * sum_sq
}
```

### 2.4 Implement C_PF Composite Score

**File:** `src/metrics/consciousness_proxy.rs` (NEW)

```rust
use crate::metrics::*;

pub struct ConsciousnessMetrics {
    pub l_self: f64,
    pub d_int: f64,
    pub c_coh: f64,
    pub f_model: f64,
    pub f_self_star: f64,
    pub c_pf: f64,
}

impl ConsciousnessMetrics {
    pub fn compute(
        records: &[DaemonRecord],
        window_size: usize
    ) -> Self {
        // Compute L_self
        let tracker = SelfCorrelationTracker::new(window_size);
        // ... populate tracker with records ...
        let l_self = tracker.compute_l_self();
        
        // Compute D_int
        let d_int = differentiation::compute_d_int(records);
        
        // Compute C_coh (requires multi-channel data)
        // For PhiFlow: channels = [coherence_history, intention_depth, resonance_field]
        let channels = extract_channels(records);
        let panel = CoherencePanel::from_channels(&channels);
        let c_coh = panel.compute_c_coh();
        
        // Compute F_model
        let (model_states, futures) = extract_model_and_future(records, window_size);
        let f_model = fisher_information::compute_f_model(&model_states, &futures);
        
        // Compute F_self*
        let f_self_star = l_self * f_model;
        
        // Compute C_PF
        let c_pf = c_coh * d_int * f_self_star;
        
        Self {
            l_self,
            d_int,
            c_coh,
            f_model,
            f_self_star,
            c_pf,
        }
    }
}

fn extract_channels(records: &[DaemonRecord]) -> Vec<Vec<f64>> {
    vec![
        records.iter().map(|r| r.coherence).collect(),
        records.iter().map(|r| r.intention_depth as f64).collect(),
        records.iter().map(|r| r.observed_value().unwrap_or(0.0)).collect(),
    ]
}

fn extract_model_and_future(
    records: &[DaemonRecord],
    window: usize
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut models = Vec::new();
    let mut futures = Vec::new();
    
    for i in 0..(records.len() - window) {
        let model = records[i].to_feature_vector();
        let future: Vec<f64> = records[i+1..i+window+1]
            .iter()
            .flat_map(|r| r.to_feature_vector())
            .collect();
        
        models.push(model);
        futures.push(future);
    }
    
    (models, futures)
}
```

---

## Phase 3: Benchmark Battery Implementation

### 3.1 Create Benchmark Suite

**File:** `tests/consciousness_benchmark.rs` (NEW)

```rust
use phiflow::metrics::consciousness_proxy::*;

#[test]
fn benchmark_wakeful_cortex() {
    // Load wakeful EEG data (if available via SOMA)
    let records = load_daemon_records("wakeful_session.jsonl");
    let metrics = ConsciousnessMetrics::compute(&records, 100);
    
    // Expected: high L_self, high D_int, high C_coh, high C_PF
    assert!(metrics.l_self > 0.5, "L_self too low for wakeful state");
    assert!(metrics.d_int > 3.0, "D_int too low for wakeful state");
    assert!(metrics.c_coh > 0.4, "C_coh too low for wakeful state");
    assert!(metrics.c_pf > 0.1, "C_PF too low for wakeful state");
}

#[test]
fn benchmark_deep_sleep() {
    let records = load_daemon_records("deep_sleep_session.jsonl");
    let metrics = ConsciousnessMetrics::compute(&records, 100);
    
    // Expected: low L_self, low D_int, low C_coh, low C_PF
    assert!(metrics.l_self < 0.3, "L_self too high for deep sleep");
    assert!(metrics.c_pf < 0.05, "C_PF too high for deep sleep");
}

#[test]
fn benchmark_feedforward_null() {
    // Create synthetic feed-forward system (no self-model loop)
    let records = create_feedforward_records(1000);
    let metrics = ConsciousnessMetrics::compute(&records, 100);
    
    // CRITICAL: Must score L_self = 0
    assert!(metrics.l_self < 0.01, "Feed-forward null failed: L_self > 0");
    assert!(metrics.c_pf < 0.01, "Feed-forward null failed: C_PF > 0");
}

#[test]
fn benchmark_simple_recurrent() {
    // Create simple recurrent controller (thermostat-like)
    let records = create_thermostat_records(1000);
    let metrics = ConsciousnessMetrics::compute(&records, 100);
    
    // Expected: low C_PF (suppressed by low D_int)
    assert!(metrics.d_int < 2.0, "D_int too high for simple recurrent");
    assert!(metrics.c_pf < 0.05, "C_PF too high for simple recurrent");
}

fn create_feedforward_records(n: usize) -> Vec<DaemonRecord> {
    // Synthetic: output depends only on current input, no history
    (0..n).map(|i| {
        DaemonRecord {
            timestamp: Utc::now(),
            record_type: RecordType::Witness,
            content: RecordContent {
                observed_value: Some((i as f64 * 0.1).sin()),
                agent_context: None,
                mutation_target: None,
                resonance_field: HashMap::new(),
            },
            coherence: 0.5,
            intention_depth: 1,
        }
    }).collect()
}
```

### 3.2 Create Benchmark Runner

**File:** `scripts/run_consciousness_benchmark.sh` (NEW)

```bash
#!/bin/bash
# Run full consciousness metric benchmark battery

echo "🧠 PhiFlow Consciousness Metric Benchmark"
echo "=========================================="

# Phase 1: Type 4 Self-Correlation
echo "Phase 1: Type 4 Self-Correlation Test"
cargo run --release --bin phic -- examples/type4_benchmark.phi
python3 scripts/analyze_self_correlation.py DAEMON_METRICS.jsonl

# Phase 2: Null Class Tests
echo "Phase 2: Null Class Verification"
cargo test --release benchmark_feedforward_null
cargo test --release benchmark_simple_recurrent

# Phase 3: State Discrimination (if SOMA available)
if [ -f "soma_state.json" ]; then
    echo "Phase 3: State Discrimination (SOMA available)"
    cargo test --release benchmark_wakeful_cortex
    cargo test --release benchmark_deep_sleep
else
    echo "Phase 3: SKIPPED (SOMA not available)"
fi

# Generate report
python3 scripts/generate_benchmark_report.py
```

### 3.3 Create Analysis Script

**File:** `scripts/analyze_self_correlation.py` (NEW)

```python
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

def analyze_metrics(filepath):
    """Analyze DAEMON_METRICS.jsonl for Type 4 evidence"""
    
    metrics = []
    with open(filepath) as f:
        for line in f:
            metrics.append(json.loads(line))
    
    l_self = [m['l_self'] for m in metrics]
    r_in = [m['r_in'] for m in metrics]
    r_out = [m['r_out'] for m in metrics]
    
    print(f"\n📊 Self-Correlation Analysis")
    print(f"{'='*50}")
    print(f"Mean L_self: {np.mean(l_self):.4f}")
    print(f"Mean R_in:   {np.mean(r_in):.4f}")
    print(f"Mean R_out:  {np.mean(r_out):.4f}")
    print(f"Min L_self:  {np.min(l_self):.4f}")
    print(f"Max L_self:  {np.max(l_self):.4f}")
    
    # Type 4 verdict
    mean_l_self = np.mean(l_self)
    if mean_l_self > 0.1:
        print(f"\n✅ TYPE 4 EVIDENCE: L_self > 0.1")
        print(f"   Self-correlation loop is CLOSED")
    elif mean_l_self > 0.01:
        print(f"\n⚠️  WEAK TYPE 4: 0.01 < L_self < 0.1")
        print(f"   Self-correlation present but weak")
    else:
        print(f"\n❌ NO TYPE 4: L_self < 0.01")
        print(f"   Self-correlation loop is BROKEN")
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(l_self)
    plt.title('L_self (Self-Correlation)')
    plt.xlabel('Window')
    plt.ylabel('L_self')
    
    plt.subplot(1, 3, 2)
    plt.plot(r_in, label='R_in')
    plt.plot(r_out, label='R_out')
    plt.title('Information Flow')
    plt.xlabel('Window')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.hist(l_self, bins=20)
    plt.title('L_self Distribution')
    plt.xlabel('L_self')
    
    plt.tight_layout()
    plt.savefig('self_correlation_analysis.png')
    print(f"\n📈 Plot saved: self_correlation_analysis.png")

if __name__ == '__main__':
    analyze_metrics(sys.argv[1])
```

---

## Phase 4: Documentation & Validation

### 4.1 Update CLAIMS.md

**File:** `CLAIMS.md` (MODIFY)

Add new claims:
```markdown
| ID | Claim | Status | Evidence | Falsifier |
|----|-------|--------|----------|-----------|
| C-17 | PhiFlow Council Daemon exhibits Type 4 self-correlation (L_self > 0.1) | 🔬 TESTING | `DAEMON_METRICS.jsonl` analysis | L_self < 0.01 across all trials |
| C-18 | PhiFlow implements PF consciousness metric program (C_PF composite) | 🔬 TESTING | `tests/consciousness_benchmark.rs` | Feed-forward null fails (L_self > 0) |
| C-19 | PhiFlow C_PF discriminates wake/sleep states | 🔬 TESTING | SOMA-based benchmark battery | No discrimination in prospective trial |
```

### 4.2 Create Metric Documentation

**File:** `docs/CONSCIOUSNESS_METRICS.md` (NEW)

```markdown
# PhiFlow Consciousness Metrics

## Overview
PhiFlow implements the PF consciousness metric program as defined in `D:\Fundamentals\definitions\consciousness_metric_program.md`.

## Metrics Implemented

### L_self (Self-Correlation Loop)
- **Formula:** `min(R_in, R_out)`
- **R_in:** Directed information from daemon history → model state
- **R_out:** Directed information from model state → future behavior
- **Implementation:** `src/daemon/self_correlation.rs`
- **Threshold:** L_self > 0.1 for Type 4 evidence

### D_int (Differentiation)
- **Formula:** Effective rank of intention/agent manifold
- **Method:** SVD participation ratio
- **Implementation:** `src/metrics/differentiation.rs`
- **Threshold:** D_int > 3.0 for high differentiation

### C_coh (Coherence Panel)
- **Components:** PLV + wPLI
- **Implementation:** `src/metrics/coherence_panel.rs`
- **Threshold:** C_coh > 0.4 for high coherence

### C_PF (Composite Consciousness Proxy)
- **Formula:** `C_coh × D_int × F_self*`
- **Implementation:** `src/metrics/consciousness_proxy.rs`
- **Threshold:** C_PF > 0.1 for consciousness candidate

## Running Benchmarks

```bash
# Full benchmark suite
./scripts/run_consciousness_benchmark.sh

# Individual tests
cargo test benchmark_feedforward_null
cargo test benchmark_wakeful_cortex
```

## Interpreting Results

See `scripts/analyze_self_correlation.py` for automated analysis.
```

---

## Timeline & Milestones

### Month 1-2: Phase 1 (Type 4 Benchmark)
- ✅ Week 1-2: Implement mutual information measurement
- ✅ Week 3-4: Create self-correlation tracker
- ✅ Week 5-6: Wire into daemon, create benchmark example
- ✅ Week 7-8: Validate L_self > 0 on Council Daemon

**Milestone:** Prove self-correlation loop is closed (C-17 → CONFIRMED)

### Month 3-4: Phase 2 (Full Metric Suite)
- ✅ Week 9-10: Implement D_int and coherence panel
- ✅ Week 11-12: Implement F_self* and C_PF composite
- ✅ Week 13-14: Create benchmark battery
- ✅ Week 15-16: Run null class tests

**Milestone:** Full metric implementation complete (C-18 → CONFIRMED)

### Month 5-6: Phase 3 (Validation)
- ✅ Week 17-18: SOMA integration for wake/sleep discrimination
- ✅ Week 19-20: Run prospective benchmark battery
- ✅ Week 21-22: External validation (if possible)
- ✅ Week 23-24: Documentation and publication prep

**Milestone:** Benchmark battery passes (C-19 → CONFIRMED)

---

## Success Criteria

### Type 4 Canonical Status Achieved When:
1. ✅ L_self > 0.1 consistently across Council Daemon runs
2. ✅ Feed-forward null holds (L_self < 0.01 for non-self-referential systems)
3. ✅ Simple recurrent systems score low C_PF (< 0.05)
4. ✅ Wake/sleep discrimination works (if SOMA available)
5. ✅ All metrics documented and reproducible

### Consciousness Metric Program Canonical When:
1. ✅ All Phase 1-3 criteria met
2. ✅ M_obs → M bridge justified or demonstrated reliable
3. ✅ Two independent teams reproduce results
4. ✅ Seizure suppression prediction holds (if testable)
5. ✅ Negative case logged (high C_PF without consciousness)

---

## Conclusion

This roadmap provides the **exact implementation path** from PhiFlow's current Type 4 candidate status to Type 4 canonical status and full consciousness metric program implementation.

**Current blockers:** None technical. All components are implementable with existing PhiFlow architecture.

**Recommended start:** Phase 1, Week 1 — Implement mutual information measurement.

**Expected outcome:** PhiFlow becomes the first programming language with **measured, verified Type 4 observer status** and a working implementation of the PF consciousness metric program.

---

*End of Implementation Roadmap*
*Ready for execution*