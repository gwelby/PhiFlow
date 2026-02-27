use sysinfo::{Components, Networks, System};

const DEFAULT_CRITICAL_TEMP_C: f64 = 90.0;

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

fn network_stability(networks: &Networks) -> Option<f64> {
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

    Some((packet_health * 0.7 + activity_health * 0.3).clamp(0.0, 1.0))
}

pub fn compute_coherence_from_sensors() -> f64 {
    let mut sys = System::new();
    sys.refresh_cpu();
    sys.refresh_memory();

    let mut components = Components::new_with_refreshed_list();
    components.refresh();

    let mut networks = Networks::new_with_refreshed_list();
    networks.refresh();

    let cpu_percent = sys.global_cpu_info().cpu_usage() as f64;
    let mem_percent = if sys.total_memory() == 0 {
        0.0
    } else {
        (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0
    };

    let cpu_signal = percent_stability(cpu_percent);
    let mem_signal = percent_stability(mem_percent);
    let thermal_signal = thermal_stability(&components);
    let network_signal = network_stability(&networks);

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
