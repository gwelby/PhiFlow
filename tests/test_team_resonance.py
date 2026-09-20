import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os
import sys

# Add root directory to path so team_resonance can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import team_resonance
from team_resonance import bar


class TestTeamResonance(unittest.TestCase):
    @patch('team_resonance.Path.exists')
    @patch('team_resonance.subprocess.run')
    def test_run_phi_timeout(self, mock_run, mock_exists):
        """Test that run_phi handles subprocess.TimeoutExpired correctly."""
        mock_exists.return_value = True

        # We need to simulate subprocess.TimeoutExpired, which requires cmd and timeout args
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="phic", timeout=30)

        result = team_resonance.run_phi("test_agent", "test.phi")

        # Expected: name, [], None, [], 30.0, "TIMEOUT"
        self.assertEqual(result, ("test_agent", [], None, [], 30.0, "TIMEOUT"))

    @patch('team_resonance.Path.exists')
    @patch('team_resonance.subprocess.run')
    def test_run_phi_exception(self, mock_run, mock_exists):
        """Test that run_phi handles generic Exceptions correctly."""
        mock_exists.return_value = True

        error_msg = "Something went wrong"
        mock_run.side_effect = Exception(error_msg)

        result = team_resonance.run_phi("test_agent", "test.phi")

        # Expected: name, [], None, [], 0.0, "Something went wrong"
        self.assertEqual(result, ("test_agent", [], None, [], 0.0, error_msg))

    @patch('team_resonance.Path.exists')
    @patch('team_resonance.subprocess.run')
    def test_run_phi_success(self, mock_run, mock_exists):
        """Test that run_phi correctly parses a successful execution."""
        mock_exists.return_value = True

        mock_result = MagicMock()
        mock_result.stdout = "Resonating Field: 12.34Hz\nFinal Coherence: 0.99\nWave: data\nStream broken: test_stream"
        mock_run.return_value = mock_result

        # We also need to mock time.time to get a predictable elapsed time
        with patch('team_resonance.time.time', side_effect=[100.0, 105.5]):
            result = team_resonance.run_phi("test_agent", "test.phi")

        # Expected: name, [12.34], 0.99, ["test_stream"], 5.5, None
        self.assertEqual(result, ("test_agent", [12.34], 0.99, ["test_stream"], 5.5, None))

    @patch('team_resonance.Path.exists')
    def test_run_phi_file_not_found(self, mock_exists):
        """Test that run_phi correctly handles a missing file."""
        mock_exists.return_value = False

        result = team_resonance.run_phi("test_agent", "missing.phi")

        # Expected: name, [], None, [], 0.0, "NOT FOUND"
        self.assertEqual(result, ("test_agent", [], None, [], 0.0, "NOT FOUND"))

if __name__ == '__main__':
    unittest.main()


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
