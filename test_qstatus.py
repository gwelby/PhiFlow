#!/usr/bin/env python3
"""
Test script for Quantum Project Status Manager
"""

import os
import sys
import shutil
import tempfile
import unittest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class TestQuantumStatusManager(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        os.chdir(self.test_dir)

        # Create a test file
        with open('test.txt', 'w') as f:
            f.write('Test file')

        # Initialize a git repo
        os.system('git init')

    def tearDown(self):
        """Clean up test environment"""
        os.chdir(os.path.expanduser('~'))  # Move out of test dir
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_show_status_command(self):
        """Test the show status command"""
        from click.testing import CliRunner
        from src.quantum_project_manager.qstatus import cli

        runner = CliRunner()

        # Create a basic STATUS.md file
        with open('STATUS.md', 'w', encoding='utf-8') as f:
            f.write("""---
project: 'Test Project'
status: active
frequency: 432.0
coherence: 0.8
phi_harmonic: ground
last_updated: '2025-05-22T00:00:00'
---
# Test Project Status
""")

        result = runner.invoke(cli, ['show'])
        self.assertEqual(result.exit_code, 0)
        # The show command will regenerate the status, so we just check for the title
        self.assertIn('Quantum Project Status', result.output)

    def test_add_component_command(self):
        """Test adding a component"""
        from click.testing import CliRunner
        from src.quantum_project_manager.qstatus import cli

        runner = CliRunner()

        # Create a basic STATUS.md file
        with open('STATUS.md', 'w') as f:
            f.write("""---
project: 'Test Project'
status: active
frequency: 432.0
coherence: 0.8
phi_harmonic: ground
last_updated: '2025-05-22T00:00:00'
components: []
---
# Test Project Status
""")

        # Add a component
        result = runner.invoke(cli, [
            'add-component', 'Test Component',
            '--status', 'active',
            '--owner', 'Tester',
            '--frequency', '432'
        ])

        self.assertEqual(result.exit_code, 0)

        # Verify the component was added
        with open('STATUS.md', 'r') as f:
            content = f.read()
            self.assertIn('Test Component', content)

    def test_calibrate_command(self):
        """Test quantum state calibration"""
        from click.testing import CliRunner
        from src.quantum_project_manager.qstatus import cli
        import time

        runner = CliRunner()

        # Create a basic STATUS.md file
        with open('STATUS.md', 'w', encoding='utf-8') as f:
            f.write("""---
project: 'Test Project'
status: active
frequency: 432.0
coherence: 0.8
phi_harmonic: ground
last_updated: '2025-05-22T00:00:00'
---
# Test Project Status
""")

        # Calibrate quantum state
        result = runner.invoke(cli, [
            'calibrate',
            '--frequency', '528',
            '--coherence', '0.9',
            '--harmonic', 'create'
        ])

        self.assertEqual(result.exit_code, 0)

        # Small delay to ensure file is written
        time.sleep(0.5)

        # Verify the calibration by checking the file directly
        with open('STATUS.md', 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn('frequency: 528.0', content)
            self.assertIn('coherence: 0.9', content)
            self.assertIn('phi_harmonic: create', content)

if __name__ == '__main__':
    unittest.main()
