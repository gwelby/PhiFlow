from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Optional

import psutil


@dataclass
class P1SensorReading:
    cpu_percent: float
    memory_percent: float
    cpu_temp_celsius: Optional[float]
    timestamp_utc: str


def _read_cpu_temp_celsius() -> Optional[float]:
    """Read CPU temperature from psutil when available."""
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        logging.warning("thermal sensor unavailable on platform")
        return None

    if not temps:
        logging.warning("thermal sensor unavailable on platform")
        return None

    for entries in temps.values():
        for entry in entries:
            current = getattr(entry, "current", None)
            if current is not None:
                return float(current)

    logging.warning("thermal sensor unavailable on platform")
    return None


def read_sensors() -> P1SensorReading:
    """Read live P1 sensor data using psutil."""
    cpu_percent = float(psutil.cpu_percent(interval=0.2))
    if cpu_percent == 0.0:
        cpu_percent = float(psutil.cpu_percent(interval=0.4))

    memory_percent = float(psutil.virtual_memory().percent)
    cpu_temp_celsius = _read_cpu_temp_celsius()

    return P1SensorReading(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        cpu_temp_celsius=cpu_temp_celsius,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )