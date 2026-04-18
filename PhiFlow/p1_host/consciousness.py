from __future__ import annotations

from .sensors import P1SensorReading


def compute_coherence(reading: P1SensorReading) -> float:
    """Derive 0.0-1.0 coherence score from real sensor state."""
    cpu_stability = 1.0 - (reading.cpu_percent / 100.0)
    mem_stability = 1.0 - (reading.memory_percent / 100.0)
    base_coherence = (cpu_stability * 0.5) + (mem_stability * 0.5)

    temp = reading.cpu_temp_celsius
    if temp is not None and 40.0 <= temp <= 55.0:
        thermal_factor = 1.0 - abs(temp - 47.0) / 8.0
        coherence = (base_coherence * 0.8) + (thermal_factor * 0.2)
    else:
        coherence = base_coherence

    return max(0.0, min(1.0, float(coherence)))