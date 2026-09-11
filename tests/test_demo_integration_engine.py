import sys
import unittest
from unittest.mock import patch, MagicMock, call
import time

# Add the project root to python path so demo_integration_engine can be found
sys.path.append('.')
sys.path.append('src')

# Due to sys.modules patching required before demo_integration_engine imports phi_flow_integration_engine
sys.modules['numpy'] = MagicMock()

import demo_integration_engine
from src.integration.phi_flow_integration_engine import (
    OptimizationLevel,
    ConsciousnessState
)

class TestDemoIntegrationEngine(unittest.TestCase):

    def setUp(self):
        # We need numpy mocked for any other modules importing this one
        pass

    def tearDown(self):
        pass

    @patch('sys.stdout')
    def test_print_banner(self, mock_stdout):
        demo_integration_engine.print_banner()
        self.assertTrue(mock_stdout.write.called)

    @patch('sys.stdout')
    def test_print_sacred_mathematics(self, mock_stdout):
        demo_integration_engine.print_sacred_mathematics()
        self.assertTrue(mock_stdout.write.called)

    @patch('sys.stdout')
    def test_demonstrate_optimization_levels(self, mock_stdout):
        mock_engine = MagicMock()
        mock_engine.execute_program.return_value = {
            'success': True,
            'performance': {
                'speedup_achieved': 1.5,
                'phi_efficiency': 0.9,
                'coherence_maintained': 0.99
            }
        }

        demo_integration_engine.demonstrate_optimization_levels(mock_engine)

        self.assertEqual(mock_engine.execute_program.call_count, len(OptimizationLevel))
        self.assertTrue(mock_stdout.write.called)

        # Test failure path
        mock_engine.execute_program.return_value = {
            'success': False,
            'error': 'Test error'
        }
        demo_integration_engine.demonstrate_optimization_levels(mock_engine)
        self.assertTrue(mock_stdout.write.called)

    @patch('sys.stdout')
    def test_demonstrate_consciousness_optimization(self, mock_stdout):
        mock_engine = MagicMock()
        mock_engine.optimize_consciousness_state.return_value = {
            'target_frequency_hz': 432.0,
            'frequency_alignment': 0.95,
            'coherence_before': 0.8,
            'coherence_after': 0.95
        }

        demo_integration_engine.demonstrate_consciousness_optimization(mock_engine)

        self.assertEqual(mock_engine.optimize_consciousness_state.call_count, len(ConsciousnessState))
        self.assertTrue(mock_stdout.write.called)

    @patch('sys.stdout')
    def test_demonstrate_complex_execution(self, mock_stdout):
        mock_engine = MagicMock()
        mock_engine.execute_program.return_value = {
            'success': True,
            'performance': {
                'speedup_achieved': 2.0,
                'coherence_maintained': 0.99,
                'consciousness_enhancement': 1.2,
                'frequency_alignment': 0.98,
                'total_duration_seconds': 0.5
            },
            'phases': {
                'health_check': 0.1,
                'execution': 0.4
            }
        }

        demo_integration_engine.demonstrate_complex_execution(mock_engine)

        mock_engine.execute_program.assert_called_once()
        self.assertTrue(mock_stdout.write.called)

        # Test failure path
        mock_engine.execute_program.return_value = {
            'success': False,
            'error': 'Test complex error'
        }
        demo_integration_engine.demonstrate_complex_execution(mock_engine)

    @patch('sys.stdout')
    def test_demonstrate_performance_analytics(self, mock_stdout):
        mock_engine = MagicMock()

        # Test with executions
        mock_engine.get_performance_analytics.return_value = {
            'total_executions': 5,
            'success_rate': 0.8,
            'successful_executions': 4,
            'averages': {
                'duration_seconds': 1.0,
                'speedup_achieved': 1.5,
                'coherence_maintained': 0.9,
                'consciousness_enhancement': 1.1
            },
            'recent_executions': [
                {'success': True, 'execution_id': '1', 'duration': 1.0, 'speedup': 1.5},
                {'success': False, 'execution_id': '2', 'duration': 0.5, 'speedup': 1.0}
            ]
        }

        demo_integration_engine.demonstrate_performance_analytics(mock_engine)
        mock_engine.get_performance_analytics.assert_called_once()
        self.assertTrue(mock_stdout.write.called)

        # Test without successful executions
        mock_engine.get_performance_analytics.return_value = {
            'total_executions': 1,
            'success_rate': 0.0,
            'successful_executions': 0
        }
        demo_integration_engine.demonstrate_performance_analytics(mock_engine)

    @patch('sys.stdout')
    def test_demonstrate_system_health(self, mock_stdout):
        mock_engine = MagicMock()

        mock_health = MagicMock()
        mock_health.overall_health = 0.95
        mock_health.coherence_engine_status = True
        mock_health.optimizer_status = True
        mock_health.parser_status = True
        mock_health.consciousness_monitor_status = True
        mock_health.cuda_status = False
        mock_health.memory_available_gb = 16.0
        mock_health.cpu_usage_percent = 25.0

        mock_engine.get_health_status.return_value = mock_health

        demo_integration_engine.demonstrate_system_health(mock_engine)
        mock_engine.get_health_status.assert_called_once()
        self.assertTrue(mock_stdout.write.called)

    @patch('time.sleep')
    @patch('demo_integration_engine.demonstrate_system_health')
    @patch('demo_integration_engine.demonstrate_optimization_levels')
    @patch('demo_integration_engine.demonstrate_consciousness_optimization')
    @patch('demo_integration_engine.demonstrate_complex_execution')
    @patch('demo_integration_engine.demonstrate_performance_analytics')
    @patch('demo_integration_engine.PhiFlowIntegrationEngine')
    @patch('sys.stdout')
    def test_main(self, mock_stdout, mock_engine_cls, mock_demo_perf, mock_demo_comp, mock_demo_cons, mock_demo_opt, mock_demo_health, mock_sleep):
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance

        demo_integration_engine.main()

        mock_engine_cls.assert_called_once()
        mock_demo_health.assert_called_once_with(mock_engine_instance)
        mock_demo_opt.assert_called_once_with(mock_engine_instance)
        mock_demo_cons.assert_called_once_with(mock_engine_instance)
        mock_demo_comp.assert_called_once_with(mock_engine_instance)
        mock_demo_perf.assert_called_once_with(mock_engine_instance)
        mock_engine_instance.shutdown.assert_called_once()

    @patch('demo_integration_engine.PhiFlowIntegrationEngine')
    @patch('sys.stdout')
    def test_main_exception(self, mock_stdout, mock_engine_cls):
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance

        # Test exception during demonstration functions
        with patch('demo_integration_engine.demonstrate_system_health', side_effect=Exception("Test Exception")):
            demo_integration_engine.main()

        self.assertTrue(mock_stdout.write.called)
        mock_engine_instance.shutdown.assert_called_once()

# Clean up sys.modules after this module is imported to avoid polluting other tests
if 'numpy' in sys.modules and isinstance(sys.modules['numpy'], MagicMock):
    del sys.modules['numpy']

if __name__ == '__main__':
    unittest.main()
