import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os
import sys

# Add root directory to path so team_resonance can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import team_resonance

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
