import pytest
import sys
import os

# Add root directory to path to allow importing team_resonance
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from team_resonance import bar

def test_bar_none():
    assert bar(None) == "──────────────────────  (no signal)"
    assert bar(None, width=10) == "──────────  (no signal)"

def test_bar_standard_values():
    # width=22 default
    # 0.0 -> scaled = 0.0, filled = 0
    assert bar(0.0) == "░░░░░░░░░░░░░░░░░░░░░░  0.0000 Hz"

    # 0.5 -> scaled = 0.5, filled = 11
    assert bar(0.5) == "███████████░░░░░░░░░░░  0.5000 Hz"

    # 1.0 -> scaled = 1.0, filled = 22
    assert bar(1.0) == "██████████████████████  1.0000 Hz"

    # 2.0 -> scaled = 1.0, filled = 22
    assert bar(2.0) == "██████████████████████  2.0000 Hz"

def test_bar_scaled_values():
    # > 2.0 values
    # scaled = min(value / 500.0, 1.0)

    # 250.0 -> scaled = 0.5, filled = 11
    assert bar(250.0) == "███████████░░░░░░░░░░░  250.0000 Hz  (scaled)"

    # 432.0 -> scaled = 0.864, filled = 19 (0.864 * 22 = 19.008 -> 19)
    assert bar(432.0) == "███████████████████░░░  432.0000 Hz  (scaled)"

    # 500.0 -> scaled = 1.0, filled = 22
    assert bar(500.0) == "██████████████████████  500.0000 Hz  (scaled)"

    # 1000.0 -> scaled = 1.0, filled = 22
    assert bar(1000.0) == "██████████████████████  1000.0000 Hz  (scaled)"

def test_bar_edge_cases():
    # Negative values
    # max(-1.0, 0.0) -> 0.0, filled = 0
    assert bar(-1.0) == "░░░░░░░░░░░░░░░░░░░░░░  -1.0000 Hz"

def test_bar_custom_width():
    # width=10
    assert bar(0.5, width=10) == "█████░░░░░  0.5000 Hz"
    assert bar(250.0, width=10) == "█████░░░░░  250.0000 Hz  (scaled)"
