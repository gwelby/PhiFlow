use sysinfo::System;

pub fn compute_coherence_from_sensors() -> f64 {
    let mut sys = System::new();
    sys.refresh_cpu();
    sys.refresh_memory();

    let cpu = sys.global_cpu_info().cpu_usage() as f64;
    let mem = if sys.total_memory() == 0 {
        0.0
    } else {
        (sys.used_memory() as f64 / sys.total_memory() as f64) * 100.0
    };

    let cpu_stability = 1.0 - (cpu / 100.0);
    let mem_stability = 1.0 - (mem / 100.0);
    (cpu_stability * 0.5 + mem_stability * 0.5).clamp(0.0, 1.0)
}
