import sys
import os
import pytest

# Add src to path so integration tests can import from src/integration/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Constants needed by test_sacred_mathematics.py
PHI = 1.618033988749895
GOLDEN_ANGLE = 137.50776405003785
SACRED_FREQUENCIES = [432, 528, 594, 672, 720, 768, 963]
CONSCIOUSNESS_STATES = ["OBSERVE", "CREATE", "INTEGRATE", "HARMONIZE", "TRANSCEND", "CASCADE", "SUPERPOSITION"]


@pytest.fixture
def phi_constants():
    """Phi-harmonic constants for sacred math tests"""
    return {
        'PHI': PHI,
        'GOLDEN_ANGLE': GOLDEN_ANGLE,
        'SACRED_FREQUENCIES': SACRED_FREQUENCIES,
        'PHI_SQUARED': PHI ** 2,
        'PHI_CUBED': PHI ** 3,
        'PHI_FOURTH': PHI ** 4,
        'PHI_PHI': PHI ** PHI,
        'FIBONACCI_SEQUENCE': [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144],
        'GOLDEN_RATIO_CONJUGATE': 1 / PHI,
        'CONSCIOUSNESS_STATES': CONSCIOUSNESS_STATES
    }

@pytest.fixture
def performance_timer():
    """Enhanced timer for performance testing"""
    import time
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.lap_times = []
            self.markers = {}

        def start(self):
            self.start_time = time.time()
            return self

        def stop(self):
            self.end_time = time.time()
            return self.elapsed

        def lap(self, name=None):
            current_time = time.time()
            if self.start_time:
                lap_time = current_time - self.start_time
                self.lap_times.append(lap_time)
                if name:
                    self.markers[name] = lap_time
                return lap_time
            return None

        def mark(self, name):
            if self.start_time:
                self.markers[name] = time.time() - self.start_time

        @property
        def elapsed(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            elif self.start_time:
                return time.time() - self.start_time
            return None

    return PerformanceTimer()
