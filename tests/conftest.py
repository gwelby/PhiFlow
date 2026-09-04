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
