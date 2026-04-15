use crate::phi_ir::SensorKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, OnceLock, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use sysinfo::{Components, Networks, System, MINIMUM_CPU_UPDATE_INTERVAL};

const DEFAULT_CRITICAL_TEMP_C: f64 = 90.0;
const SOMA_STATE_PATH: &str = "D:/Projects/PhiHarmonic/SOMA/soma_state.json";
const SOMA_FRESHNESS_THRESHOLD_MS: u64 = 5000; // 5 second stale threshold

fn parse_updated_at_timestamp(updated_at: &str) -> Option<std::time::SystemTime> {
    // Try ISO 8601 format first
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(updated_at) {
        return Some(dt.into());
    }
    // Try common alternative format
    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(updated_at, "%Y-%m-%dT%H:%M:%S%.f") {
        Some(dt.and_utc().into())
    } else {
        None
    }
}

pub fn is_soma_state_fresh(state: &SomaState) -> bool {
    // Use health.fresh if available
    if state.health.fresh {
        return true;
    }

    // Check age against freshness threshold
    if let Some(updated_time) = parse_updated_at_timestamp(&state.updated_at) {
        if let Ok(age) = updated_time.elapsed() {
            return (age.as_millis() as u64) < SOMA_FRESHNESS_THRESHOLD_MS;
        }
    }

    // Fall back to health age check
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
pub struct SomaSensors {
    pub soma_schumann: f64,
    pub soma_432: f64,
    pub soma_presence: f64,
    pub soma_fan_hz: f64,
    pub soma_ac_60: f64,
    pub soma_peak_dbc: f64,
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
}

static LIVE_DATA: OnceLock<Arc<RwLock<LiveSensorData>>> = OnceLock::new();

fn get_live_data() -> Arc<RwLock<LiveSensorData>> {
    LIVE_DATA
        .get_or_init(|| {
            let initial_soma = match fs::read_to_string(SOMA_STATE_PATH) {
                Ok(content) => serde_json::from_str::<SomaState>(&content).ok(),
                Err(_) => None,
            };

            let initial_data = Arc::new(RwLock::new(LiveSensorData {
                coherence: 1.0,
                cpu_usage: 0.0,
                cpu_temp: None,
                memory_usage: None,
                soma: initial_soma,
            }));

            let thread_data = Arc::clone(&initial_data);

            thread::spawn(move || {
                let mut sampler = SensorSampler::new();
                // Force first sample synchronously
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

                    // --- SOMA Bridge ---
                    let soma = match fs::read_to_string(SOMA_STATE_PATH) {
                        Ok(content) => serde_json::from_str::<SomaState>(&content).ok(),
                        Err(_) => None,
                    };

                    if let Ok(mut data) = thread_data.write() {
                        data.coherence = coherence;
                        data.cpu_usage = cpu_usage;
                        data.cpu_temp = cpu_temp;
                        data.memory_usage = memory_usage;
                        data.soma = soma;
                    }
                }
            });

            initial_data
        })
        .clone()
}

fn percent_stability(usage_percent: f64) -> f64 {
    (1.0 - usage_percent / 100.0).clamp(0.0, 1.0)
}

fn thermal_stability(components: &Components) -> Option<f64> {
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

    if temp_count == 0 {
        return None;
    }

    let average_temp = total_temp / temp_count as f64;
    let critical_temp = if critical_count > 0 {
        total_critical / critical_count as f64
    } else {
        DEFAULT_CRITICAL_TEMP_C
    };
    let critical_temp = critical_temp.max(1.0);
    Some(((critical_temp - average_temp) / critical_temp).clamp(0.0, 1.0))
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

    // Packet reliability dominates. If packet counters are unavailable/zero,
    // we fall back to a conservative "unknown but likely stable" baseline.
    let packet_health = if total_packets == 0 {
        0.85
    } else {
        let error_ratio = (total_errors as f64 / total_packets as f64).clamp(0.0, 1.0);
        1.0 - error_ratio
    };

    // Lightweight activity signal to avoid over-trusting stale/idle interfaces.
    let activity_health = if total_bytes == 0 {
        0.85
    } else {
        let normalized_activity = (total_bytes as f64 / 20_000_000.0).clamp(0.0, 1.0);
        0.5 + normalized_activity * 0.5
    };

    Some((packet_health, activity_health))
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
    get_live_data().read().unwrap().coherence
}

pub fn read_sensor(sensor: SensorKind) -> Option<f64> {
    let arc = get_live_data();
    let data = arc.read().unwrap();
    match sensor {
        SensorKind::CpuUsage => Some(data.cpu_usage),
        SensorKind::CpuTemp => data.cpu_temp,
        SensorKind::MemoryUsage => data.memory_usage,
        SensorKind::SomaSchumann => data
            .soma
            .as_ref()
            .filter(|s| is_soma_state_fresh(s))
            .map(|s| s.sensors.soma_schumann),
        SensorKind::Soma432 => data
            .soma
            .as_ref()
            .filter(|s| is_soma_state_fresh(s))
            .map(|s| s.sensors.soma_432),
        SensorKind::SomaPresence => data
            .soma
            .as_ref()
            .filter(|s| is_soma_state_fresh(s))
            .map(|s| s.sensors.soma_presence),
        SensorKind::SomaFanHz => data
            .soma
            .as_ref()
            .filter(|s| is_soma_state_fresh(s))
            .map(|s| s.sensors.soma_fan_hz),
        SensorKind::SomaAc60 => data
            .soma
            .as_ref()
            .filter(|s| is_soma_state_fresh(s))
            .map(|s| s.sensors.soma_ac_60),
        SensorKind::SomaPeakDbc => data
            .soma
            .as_ref()
            .filter(|s| is_soma_state_fresh(s))
            .map(|s| s.sensors.soma_peak_dbc),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_lowers_score() {
        // Since the live system values are non-deterministic, we test the pure scoring functions

        let idle_cpu_score = percent_stability(5.0);
        let stressed_cpu_score = percent_stability(95.0);

        assert!(
            stressed_cpu_score < idle_cpu_score,
            "Stress (95% CPU) should have a lower stability score than idle (5% CPU)"
        );

        let idle_mem_score = percent_stability(20.0);
        let stressed_mem_score = percent_stability(90.0);

        assert!(
            stressed_mem_score < idle_mem_score,
            "Stress (90% Mem) should have a lower stability score than idle (20% Mem)"
        );
    }

    #[test]
    fn test_snapshot_scoring_uses_network_and_thermal_when_present() {
        let stable = SensorSnapshot {
            cpu_usage_percent: 10.0,
            memory_usage_percent: 20.0,
            cpu_temp_c: Some(40.0),
            cpu_critical_temp_c: Some(90.0),
            network_packet_health: Some(0.98),
            network_activity_health: Some(0.95),
        };
        let stressed = SensorSnapshot {
            cpu_usage_percent: 85.0,
            memory_usage_percent: 90.0,
            cpu_temp_c: Some(82.0),
            cpu_critical_temp_c: Some(90.0),
            network_packet_health: Some(0.40),
            network_activity_health: Some(0.55),
        };

        let stable_score = compute_coherence_from_snapshot(&stable);
        let stressed_score = compute_coherence_from_snapshot(&stressed);

        assert!(stable_score > stressed_score);
    }

    #[test]
    fn test_soma_state_missing_file() {
        // When soma_state.json doesn't exist, read_sensor should return None for SOMA sensors
        let result = read_sensor(SensorKind::SomaSchumann);
        // Result may be Some or None depending on whether the file exists,
        // but it should not panic
        let _ = result;
    }

    #[test]
    fn test_soma_state_freshness_check() {
        // Test with a stale state file (updated_at in the past)
        let stale_state = SomaState {
            schema_version: "1.0".to_string(),
            updated_at: "2020-01-01T00:00:00Z".to_string(),
            session_id: "test-session".to_string(),
            runtime: SomaRuntime {
                sensor_stack: "test".to_string(),
                ring_sensor_type: "test".to_string(),
                sample_rate_hz: 100.0,
                fusion_interval_hz: 10.0,
            },
            health: SomaHealth {
                fresh: false,
                age_ms: 999999,
                baseline_locked: false,
            },
            sensors: SomaSensors {
                soma_schumann: 7.83,
                soma_432: 432.0,
                soma_presence: 1.0,
                soma_fan_hz: 50.0,
                soma_ac_60: 60.0,
                soma_peak_dbc: -30.0,
            },
        };

        // Stale state should be rejected
        assert!(!is_soma_state_fresh(&stale_state));
    }

    #[test]
    fn test_soma_state_fresh_data_accepted() {
        // Test with fresh state (health.fresh = true)
        let fresh_state = SomaState {
            schema_version: "1.0".to_string(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            session_id: "test-session".to_string(),
            runtime: SomaRuntime {
                sensor_stack: "test".to_string(),
                ring_sensor_type: "test".to_string(),
                sample_rate_hz: 100.0,
                fusion_interval_hz: 10.0,
            },
            health: SomaHealth {
                fresh: true,
                age_ms: 100,
                baseline_locked: true,
            },
            sensors: SomaSensors {
                soma_schumann: 7.83,
                soma_432: 432.0,
                soma_presence: 1.0,
                soma_fan_hz: 50.0,
                soma_ac_60: 60.0,
                soma_peak_dbc: -30.0,
            },
        };

        // Fresh state should be accepted
        assert!(is_soma_state_fresh(&fresh_state));
    }

    #[test]
    fn test_soma_sensor_values_readable() {
        // Verify that all SOMA sensor kinds can be queried without panic
        let sensors = [
            SensorKind::SomaSchumann,
            SensorKind::Soma432,
            SensorKind::SomaPresence,
            SensorKind::SomaFanHz,
            SensorKind::SomaAc60,
            SensorKind::SomaPeakDbc,
        ];

        for sensor in sensors {
            // Should not panic, may return None if file doesn't exist
            let _ = read_sensor(sensor);
        }
    }

    #[test]
    fn test_soma_state_serialization_roundtrip() {
        let state = SomaState {
            schema_version: "1.0".to_string(),
            updated_at: "2026-04-14T12:00:00Z".to_string(),
            session_id: "test-123".to_string(),
            runtime: SomaRuntime {
                sensor_stack: "full".to_string(),
                ring_sensor_type: "oura".to_string(),
                sample_rate_hz: 256.0,
                fusion_interval_hz: 4.0,
            },
            health: SomaHealth {
                fresh: true,
                age_ms: 500,
                baseline_locked: true,
            },
            sensors: SomaSensors {
                soma_schumann: 7.83,
                soma_432: 432.0,
                soma_presence: 0.95,
                soma_fan_hz: 49.5,
                soma_ac_60: 60.02,
                soma_peak_dbc: -28.5,
            },
        };

        let json = serde_json::to_string(&state).expect("serialization failed");
        let deserialized: SomaState = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(deserialized.schema_version, state.schema_version);
        assert_eq!(deserialized.sensors.soma_schumann, state.sensors.soma_schumann);
        assert_eq!(deserialized.sensors.soma_432, state.sensors.soma_432);
        assert_eq!(deserialized.sensors.soma_presence, state.sensors.soma_presence);
        assert_eq!(deserialized.sensors.soma_fan_hz, state.sensors.soma_fan_hz);
        assert_eq!(deserialized.sensors.soma_ac_60, state.sensors.soma_ac_60);
        assert_eq!(deserialized.sensors.soma_peak_dbc, state.sensors.soma_peak_dbc);
    }
}
