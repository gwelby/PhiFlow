use crate::phi_ir::SensorKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use sysinfo::{Components, Networks, System, MINIMUM_CPU_UPDATE_INTERVAL};

const DEFAULT_CRITICAL_TEMP_C: f64 = 90.0;

/// Returns the base phiflow data directory using the XDG_DATA_HOME → HOME → /tmp fallback chain.
/// All runtime data paths should be derived from this to ensure consistent behaviour across
/// environments (Linux, macOS, CI, containers).
pub fn get_phiflow_data_dir() -> std::path::PathBuf {
    let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
        std::env::var("HOME")
            .map(|h| format!("{}/.local/share", h))
            .unwrap_or_else(|_| "/tmp".to_string())
    });
    let dir = std::path::PathBuf::from(base).join("phiflow");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("warning: could not create phiflow data directory {:?}: {}", dir, e);
    }
    dir
}

pub fn get_soma_state_path() -> std::path::PathBuf {
    std::env::var("SOMA_STATE_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| get_phiflow_data_dir().join("soma_state.json"))
}

pub fn get_quantum_state_path() -> std::path::PathBuf {
    std::env::var("PHIFLOW_QUANTUM_STATE_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| get_phiflow_data_dir().join("quantum_state.json"))
}

const SOMA_FRESHNESS_THRESHOLD_MS: u64 = 5000;

fn parse_updated_at_timestamp(updated_at: &str) -> Option<std::time::SystemTime> {
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(updated_at) {
        return Some(dt.into());
    }
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(updated_at, "%Y-%m-%dT%H:%M:%S%.f") {
        Some(dt.and_utc().into())
    } else {
        None
    }
}

pub fn is_soma_state_fresh(state: &SomaState) -> bool {
    if state.schema_version != "soma.phiflow.v1" && state.schema_version != "1.0" {
        return false;
    }
    if state.health.fresh {
        return true;
    }
    if let Some(updated_time) = parse_updated_at_timestamp(&state.updated_at) {
        if let Ok(age) = updated_time.elapsed() {
            return (age.as_millis() as u64) < SOMA_FRESHNESS_THRESHOLD_MS;
        }
    }
    state.health.age_ms < SOMA_FRESHNESS_THRESHOLD_MS
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaRuntime {
    pub sensor_stack: String,
    pub ring_sensor_type: String,
    pub sample_rate_hz: f64,
    pub fusion_interval_hz: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaHealth {
    pub fresh: bool,
    pub age_ms: u64,
    pub baseline_locked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SomaSensors {
    pub soma_schumann: f64,
    pub soma_432: f64,
    pub soma_presence: f64,
    pub soma_fan_hz: f64,
    pub soma_ac_60: f64,
    pub soma_peak_dbc: f64,
    pub ring_slope_1f: f64,
    pub ring_jitter_ns: f64,
    pub ring_coherence_432: f64,
    pub ring_coherence_528: f64,
    pub ring_phase_delta: f64,
}

impl Default for SomaSensors {
    fn default() -> Self {
        Self {
            soma_schumann: 0.0,
            soma_432: 0.0,
            soma_presence: 0.0,
            soma_fan_hz: 0.0,
            soma_ac_60: 0.0,
            soma_peak_dbc: 0.0,
            ring_slope_1f: 0.0,
            ring_jitter_ns: 0.0,
            ring_coherence_432: 0.0,
            ring_coherence_528: 0.0,
            ring_phase_delta: 0.0,
        }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SomaState {
    pub schema_version: String,
    pub updated_at: String,
    pub session_id: String,
    pub runtime: SomaRuntime,
    pub health: SomaHealth,
    pub sensors: SomaSensors,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensorSnapshot {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub cpu_temp_c: Option<f64>,
    pub cpu_critical_temp_c: Option<f64>,
    pub network_packet_health: Option<f64>,
    pub network_activity_health: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    pub quantum_t1: f64,
    pub quantum_t2: f64,
    pub quantum_readout_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub status: String,
    pub backend: String,
    pub updated_at: String,
    pub metrics: Option<QuantumMetrics>,
    pub error: Option<String>,
}

struct SensorSampler {
    system: System,
    components: Components,
    networks: Networks,
    last_sample_at: Option<Instant>,
    cpu_primed: bool,
}

impl SensorSampler {
    fn new() -> Self {
        Self {
            system: System::new(),
            components: Components::new_with_refreshed_list(),
            networks: Networks::new_with_refreshed_list(),
            last_sample_at: None,
            cpu_primed: false,
        }
    }

    fn sample(&mut self) -> f64 {
        if let Some(last_sample_at) = self.last_sample_at {
            let elapsed = last_sample_at.elapsed();
            if elapsed < MINIMUM_CPU_UPDATE_INTERVAL {
                thread::sleep(MINIMUM_CPU_UPDATE_INTERVAL - elapsed);
            }
        }

        if self.cpu_primed {
            self.system.refresh_cpu_usage();
        } else {
            self.system.refresh_cpu_usage();
            thread::sleep(MINIMUM_CPU_UPDATE_INTERVAL);
            self.system.refresh_cpu_usage();
            self.cpu_primed = true;
        }

        self.system.refresh_memory();
        self.components.refresh();
        self.networks.refresh();

        let coherence = compute_from_snapshot(&self.system, &self.components, &self.networks);
        self.last_sample_at = Some(Instant::now());
        coherence
    }
}

pub struct LiveSensorData {
    pub coherence: f64,
    pub cpu_usage: f64,
    pub cpu_temp: Option<f64>,
    pub memory_usage: Option<f64>,
    pub soma: Option<SomaState>,
    pub quantum: Option<QuantumState>,
}

static LIVE_DATA: OnceLock<Arc<RwLock<LiveSensorData>>> = OnceLock::new();

fn get_live_data() -> Arc<RwLock<LiveSensorData>> {
    LIVE_DATA
        .get_or_init(|| {
            let initial_soma = match fs::read_to_string(get_soma_state_path()) {
                Ok(content) => serde_json::from_str::<SomaState>(&content).ok(),
                Err(_) => None,
            };

            let initial_data = Arc::new(RwLock::new(LiveSensorData {
                coherence: 1.0,
                cpu_usage: 0.0,
                cpu_temp: None,
                memory_usage: None,
                soma: initial_soma,
                quantum: None,
            }));

            let thread_data = Arc::clone(&initial_data);

            thread::spawn(move || {
                let mut sampler = SensorSampler::new();
                let mut first_coherence = sampler.sample();

                loop {
                    let coherence = first_coherence;
                    first_coherence = sampler.sample();

                    let cpu_usage = sampler.system.global_cpu_info().cpu_usage() as f64;

                    let mut total_temp = 0.0;
                    let mut temp_count = 0usize;
                    for component in sampler.components.iter() {
                        let temp = component.temperature() as f64;
                        if temp > 0.0 {
                            total_temp += temp;
                            temp_count += 1;
                        }
                    }
                    let cpu_temp = if temp_count > 0 {
                        Some(total_temp / temp_count as f64)
                    } else {
                        None
                    };

                    let total = sampler.system.total_memory() as f64;
                    let used = sampler.system.used_memory() as f64;
                    let memory_usage = if total > 0.0 {
                        Some(used / total * 100.0)
                    } else {
                        None
                    };

                    let soma_opt = match fs::read_to_string(get_soma_state_path()) {
                        Ok(content) => serde_json::from_str::<SomaState>(&content).ok(),
                        Err(_) => None,
                    };

                    let quantum_opt = match fs::read_to_string(get_quantum_state_path()) {
                        Ok(content) => serde_json::from_str::<QuantumState>(&content).ok(),
                        Err(_) => None,
                    };

                    if let Ok(mut data) = thread_data.write() {
                        data.coherence = coherence;
                        data.cpu_usage = cpu_usage;
                        data.cpu_temp = cpu_temp;
                        data.memory_usage = memory_usage;
                        if soma_opt.is_some() {
                            data.soma = soma_opt;
                        }
                        if quantum_opt.is_some() {
                            data.quantum = quantum_opt;
                        }
                    }
                    
                    thread::sleep(Duration::from_millis(100));
                }
            });

            initial_data
        })
        .clone()
}

fn percent_stability(usage_percent: f64) -> f64 {
    (1.0 - usage_percent / 100.0).clamp(0.0, 1.0)
}

fn snapshot_from_live(
    sys: &System,
    components: &Components,
    networks: &Networks,
) -> SensorSnapshot {
    let cpu_usage_percent = sys.global_cpu_info().cpu_usage() as f64;
    let memory_usage_percent = if sys.total_memory() == 0 {
        0.0
    } else {
        (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0
    };

    let mut total_temp = 0.0;
    let mut temp_count = 0usize;
    let mut total_critical = 0.0;
    let mut critical_count = 0usize;
    for component in components.iter() {
        let temp = component.temperature() as f64;
        if temp > 0.0 {
            total_temp += temp;
            temp_count += 1;
        }
        if let Some(critical) = component.critical() {
            let critical = critical as f64;
            if critical > 0.0 {
                total_critical += critical;
                critical_count += 1;
            }
        }
    }

    let (network_packet_health, network_activity_health) = match network_health(networks) {
        Some((packet, activity)) => (Some(packet), Some(activity)),
        None => (None, None),
    };

    SensorSnapshot {
        cpu_usage_percent,
        memory_usage_percent,
        cpu_temp_c: (temp_count > 0).then(|| total_temp / temp_count as f64),
        cpu_critical_temp_c: if critical_count > 0 {
            Some(total_critical / critical_count as f64)
        } else {
            None
        },
        network_packet_health,
        network_activity_health,
    }
}

fn network_health(networks: &Networks) -> Option<(f64, f64)> {
    let mut interface_count = 0usize;
    let mut total_packets = 0u64;
    let mut total_errors = 0u64;
    let mut total_bytes = 0u64;

    for (_name, data) in networks.iter() {
        interface_count += 1;
        total_packets += data.total_packets_received() + data.total_packets_transmitted();
        total_errors += data.total_errors_on_received() + data.total_errors_on_transmitted();
        total_bytes += data.total_received() + data.total_transmitted();
    }

    if interface_count == 0 {
        return None;
    }

    let packet_health = if total_packets == 0 {
        0.85
    } else {
        let error_ratio = (total_errors as f64 / total_packets as f64).clamp(0.0, 1.0);
        1.0 - error_ratio
    };

    let activity_health = if total_bytes == 0 {
        0.85
    } else {
        let normalized_activity = (total_bytes as f64 / 20_000_000.0).clamp(0.0, 1.0);
        0.5 + normalized_activity * 0.5
    };

    Some((packet_health, activity_health))
}

pub fn compute_coherence_from_snapshot(snapshot: &SensorSnapshot) -> f64 {
    let cpu_signal = percent_stability(snapshot.cpu_usage_percent);
    let mem_signal = percent_stability(snapshot.memory_usage_percent);
    let thermal_signal = match snapshot.cpu_temp_c {
        Some(temp) => {
            let critical = snapshot
                .cpu_critical_temp_c
                .unwrap_or(DEFAULT_CRITICAL_TEMP_C)
                .max(1.0);
            Some(((critical - temp) / critical).clamp(0.0, 1.0))
        }
        None => None,
    };
    let network_signal = match (
        snapshot.network_packet_health,
        snapshot.network_activity_health,
    ) {
        (Some(packet), Some(activity)) => Some((packet * 0.7 + activity * 0.3).clamp(0.0, 1.0)),
        _ => None,
    };

    let mut weighted = cpu_signal * 0.30 + mem_signal * 0.25;
    let mut total_weight = 0.55;

    if let Some(thermal) = thermal_signal {
        weighted += thermal * 0.25;
        total_weight += 0.25;
    }
    if let Some(network) = network_signal {
        weighted += network * 0.20;
        total_weight += 0.20;
    }

    (weighted / total_weight).clamp(0.0, 1.0)
}

fn compute_from_snapshot(sys: &System, components: &Components, networks: &Networks) -> f64 {
    let snapshot = snapshot_from_live(sys, components, networks);
    compute_coherence_from_snapshot(&snapshot)
}

pub fn compute_coherence_from_sensors() -> f64 {
    let arc = get_live_data();
    let data = arc.read().unwrap();

    // Start with the system-level coherence (CPU, memory, thermal, network).
    let mut coherence = data.coherence;
    let mut total_weight = 1.0;

    // If SOMA is running and fresh, blend its environmental sensors into the
    // coherence value. This is what makes coherence a *measurement* rather
    // than a formula — the ring oscillator detects real environmental
    // conditions (EM noise, presence, vibration) that the CPU/memory sensors
    // cannot see.
    if let Some(soma) = &data.soma {
        if is_soma_state_fresh(soma) {
            // SOMA presence (0.0–1.0): environmental stability from ring
            // oscillator timing jitter. High presence = stable environment.
            let presence = soma.sensors.soma_presence.clamp(0.0, 1.0);
            coherence += presence * 0.30;
            total_weight += 0.30;

            // Fan stability: a consistent fan speed indicates thermal
            // stability. Normalize around 50 Hz (typical fan speed) with
            // a tolerance window. Sudden changes = instability.
            let fan_hz = soma.sensors.soma_fan_hz;
            if fan_hz > 0.0 {
                let fan_deviation = (fan_hz - 50.0).abs() / 50.0;
                let fan_stability = (1.0 - fan_deviation).clamp(0.0, 1.0);
                coherence += fan_stability * 0.10;
                total_weight += 0.10;
            }

            // Peak dBc: signal strength from the ring. Higher = stronger
            // signal = more coherent environment. Normalize 0–30 dBc.
            let peak_dbc = soma.sensors.soma_peak_dbc.clamp(0.0, 30.0) / 30.0;
            coherence += peak_dbc * 0.10;
            total_weight += 0.10;
        }
    }

    coherence = (coherence / total_weight).clamp(0.0, 1.0);

    // If quantum hardware metrics are available, apply them as a
    // multiplicative modifier (hardware reality penalty).
    if let Some(q) = &data.quantum {
        if let Some(m) = &q.metrics {
            let t1_factor = (m.quantum_t1 / 200.0).clamp(0.0, 1.0);
            let t2_factor = (m.quantum_t2 / 100.0).clamp(0.0, 1.0);
            let readout_factor = (1.0 - m.quantum_readout_error * 10.0).clamp(0.0, 1.0);
            let quantum_resonance = t1_factor * 0.4 + t2_factor * 0.4 + readout_factor * 0.2;
            coherence *= quantum_resonance;
        }
    }

    coherence
}

pub fn read_sensor(sensor: SensorKind) -> Option<f64> {
    let arc = get_live_data();
    let data = arc.read().unwrap();
    match sensor {
        SensorKind::CpuUsage => Some(data.cpu_usage),
        SensorKind::CpuTemp => data.cpu_temp,
        SensorKind::MemoryUsage => data.memory_usage,
        SensorKind::SomaSchumann => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.soma_schumann),
        SensorKind::Soma432 => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.soma_432),
        SensorKind::SomaPresence => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.soma_presence),
        SensorKind::SomaFanHz => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.soma_fan_hz),
        SensorKind::SomaAc60 => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.soma_ac_60),
        SensorKind::SomaPeakDbc => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.soma_peak_dbc),
        SensorKind::RingSlope1f => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.ring_slope_1f),
        SensorKind::RingJitterNs => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.ring_jitter_ns),
        SensorKind::RingCoherence432 => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.ring_coherence_432),
        SensorKind::RingCoherence528 => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.ring_coherence_528),
        SensorKind::RingPhaseDelta => data.soma.as_ref().filter(|s| is_soma_state_fresh(s)).map(|s| s.sensors.ring_phase_delta),
        SensorKind::QuantumT1 => data.quantum.as_ref().and_then(|q| q.metrics.as_ref()).map(|m| m.quantum_t1),
        SensorKind::QuantumT2 => data.quantum.as_ref().and_then(|q| q.metrics.as_ref()).map(|m| m.quantum_t2),
        SensorKind::QuantumReadoutError => data.quantum.as_ref().and_then(|q| q.metrics.as_ref()).map(|m| m.quantum_readout_error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_lowers_score() {
        let idle_cpu_score = percent_stability(5.0);
        let stressed_cpu_score = percent_stability(95.0);
        assert!(stressed_cpu_score < idle_cpu_score);
    }
}
