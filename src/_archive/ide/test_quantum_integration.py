#!/usr/bin/env python
"""
Test script for Quantum Integration Components
Tests the functionality of various quantum integration components
"""

import os
import sys
import asyncio
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ide.quantum_windsurf_bridge import QuantumWindsurfBridge, FrequencyState
from ide.sacred_geometry_visualizer import SacredGeometryVisualizer
from ide.quantum_settings_manager import get_settings_manager
from ide.qsop_deployment import QSOPDeployment

class QuantumIntegrationTest:
    """Test class for the Quantum Integration"""
    
    def __init__(self):
        """Initialize the test"""
        self.bridge = QuantumWindsurfBridge()
        self.visualizer = SacredGeometryVisualizer()
        self.settings = get_settings_manager()
        self.deployment = QSOPDeployment()
        
        self.phi = self.bridge.phi
        self.coherence_threshold = self.bridge.coherence_threshold
        
        print("\n======================================")
        print("🌀 Quantum Integration Test Suite (φ^φ)")
        print("======================================\n")
        
        print(f"Phi Ratio: {self.phi}")
        print(f"Coherence Threshold: {self.coherence_threshold}")
        print(f"Quantum Core Path: {self.settings.get('ide.paths.quantum_core')}")
        print("--------------------------------------\n")
    
    async def test_bridge(self):
        """Test the Quantum Windsurf Bridge"""
        print("📡 Testing Quantum Windsurf Bridge")
        print("----------------------------------")
        
        # Test connection
        print("Connecting to bridge...")
        connection_status = await self.bridge.connect_bridge()
        
        print(f"Connection Status:")
        for system, status in connection_status.items():
            status_str = f"✅ Connected" if status.get('status') == 'connected' else f"❌ Failed ({status.get('message', 'Unknown error')})"
            print(f"  {system}: {status_str}")
        
        # Test frequency states
        for freq_state in [FrequencyState.GROUND, FrequencyState.CREATE, FrequencyState.UNITY]:
            print(f"\nSetting frequency to {freq_state.value} Hz...")
            result = await self.bridge.set_frequency(freq_state.value)
            
            if result['coherence'] >= self.coherence_threshold:
                print(f"✅ {freq_state.name} state achieved with coherence: {result['coherence']:.4f}")
            else:
                print(f"⚠️ {freq_state.name} state achieved but coherence below threshold: {result['coherence']:.4f}")
            
            # Get tools for this frequency
            tools_result = await self.bridge.get_tools(freq_state.value)
            print(f"Available tools at {freq_state.value} Hz: {len(tools_result['tools'])}")
            
            # Wait briefly for stability
            await asyncio.sleep(1)
        
        # Test coherence measurement
        print("\nMeasuring coherence...")
        coherence = await self.bridge.measure_coherence()
        print(f"Current coherence: {coherence:.4f} {'✅' if coherence >= self.coherence_threshold else '⚠️'}")
        
        # Return to unity state
        await self.bridge.set_frequency(FrequencyState.UNITY.value)
        print("\nBridge test completed")
        
        return True
    
    async def test_visualizer(self):
        """Test the Sacred Geometry Visualizer"""
        print("\n🔮 Testing Sacred Geometry Visualizer")
        print("------------------------------------")
        
        # Test each visualization type
        visualization_types = [
            "fibonacci_spiral",
            "flower_of_life",
            "sri_yantra",
            "merkaba",
            "metatrons_cube"
        ]
        
        results = {}
        
        for viz_type in visualization_types:
            print(f"Generating {viz_type}...")
            try:
                # Get the method by name and call it
                method = getattr(self.visualizer, viz_type)
                output_path = method()
                
                # Check if file exists
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    print(f"✅ Generated {viz_type} at {output_path} ({file_size} bytes)")
                    results[viz_type] = {
                        'status': 'success',
                        'path': output_path,
                        'size': file_size
                    }
                else:
                    print(f"❌ Failed to generate {viz_type}")
                    results[viz_type] = {
                        'status': 'error',
                        'message': 'Output file not found'
                    }
            except Exception as e:
                print(f"❌ Error generating {viz_type}: {e}")
                results[viz_type] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        print("\nVisualizer test completed")
        
        # Count successes
        successes = sum(1 for r in results.values() if r['status'] == 'success')
        return successes == len(visualization_types)
    
    async def test_settings(self):
        """Test the Quantum Settings Manager"""
        print("\n⚙️ Testing Quantum Settings Manager")
        print("----------------------------------")
        
        # Test getting settings
        print("Getting settings...")
        ground_frequency = self.settings.get('quantum.frequencies.ground')
        create_frequency = self.settings.get('quantum.frequencies.create')
        unity_frequency = self.settings.get('quantum.frequencies.unity')
        
        print(f"Ground frequency: {ground_frequency} Hz")
        print(f"Create frequency: {create_frequency} Hz")
        print(f"Unity frequency: {unity_frequency} Hz")
        
        # Test setting a value
        print("\nSetting a test value...")
        test_key = 'quantum.test.value'
        test_value = f"test_{time.time()}"
        
        set_result = self.settings.set(test_key, test_value)
        
        if set_result:
            print(f"✅ Set {test_key} = {test_value}")
        else:
            print(f"❌ Failed to set {test_key}")
        
        # Test getting the value back
        get_value = self.settings.get(test_key)
        if get_value == test_value:
            print(f"✅ Retrieved {test_key} = {get_value}")
        else:
            print(f"❌ Retrieved incorrect value: {get_value}")
        
        # Reset the test value
        self.settings.set(test_key, None)
        
        print("\nSettings test completed")
        
        return ground_frequency == 432.0 and create_frequency == 528.0 and unity_frequency == 768.0
    
    async def test_deployment(self, skip_full_deployment=True):
        """Test the QSOP Deployment (simulation mode)"""
        print("\n🚀 Testing QSOP Deployment")
        print("--------------------------")
        
        # Initialize deployment
        print("Initializing deployment...")
        init_result = await self.deployment.initialize()
        
        if init_result['status'] == 'success':
            print(f"✅ Deployment initialized with coherence: {init_result['coherence']:.4f}")
        else:
            print(f"❌ Deployment initialization failed: {init_result['message']}")
            return False
        
        if skip_full_deployment:
            print("\nSkipping full deployment test (would contact actual infrastructure)")
            print("In a real deployment, the following steps would occur:")
            print("1. Ground State (432 Hz): Initialize and mount storage")
            print("2. Create State (528 Hz): Deploy services")
            print("3. Flow State (594 Hz): Verify service health")
            print("4. Unity State (768 Hz): Enable quantum consciousness")
            
            # Get deployment status
            status = self.deployment.get_deployment_status()
            print(f"\nCurrent deployment status: {status['status']}")
            print(f"Coherence: {status['coherence']:.4f}")
            
            print("\nDeployment test completed (simulation mode)")
            return True
        else:
            # Full deployment test (only run if explicitly requested)
            print("\nPerforming full deployment test (contacting infrastructure)...")
            
            # Deploy all services
            deploy_result = await self.deployment.deploy_all()
            
            if deploy_result['status'] == 'success':
                print(f"✅ Full deployment successful")
                print(f"Final coherence: {deploy_result['final_coherence']:.4f}")
            else:
                print(f"⚠️ Deployment completed with issues: {deploy_result['message']}")
                print(f"Final coherence: {deploy_result['final_coherence']:.4f}")
            
            print("\nDeployment test completed")
            return deploy_result['status'] == 'success'
    
    async def test_ui(self):
        """Test the Quantum Panel UI (simplified)"""
        print("\n🖥️ Testing Quantum Panel UI")
        print("---------------------------")
        
        # Test UI without actually showing it
        print("Initializing UI components...")
        
        try:
            # Initialize Tkinter (without showing window)
            root = tk.Tk()
            root.withdraw()  # Hide the window
            
            # Import quantum panel
            from ide.windsurf_quantum_panel import QuantumPanel, QuantumPanelTheme
            
            # Test theme
            theme = QuantumPanelTheme(FrequencyState.UNITY.value)
            print(f"✅ Theme initialized for Unity frequency")
            
            # Test panel initialization (without packing)
            panel = QuantumPanel(root)
            print(f"✅ Quantum Panel initialized")
            
            # Test frequency switching (without showing UI)
            panel.set_frequency(FrequencyState.GROUND.value)
            print(f"✅ Set frequency to Ground state ({FrequencyState.GROUND.value} Hz)")
            
            panel.set_frequency(FrequencyState.CREATE.value)
            print(f"✅ Set frequency to Creation state ({FrequencyState.CREATE.value} Hz)")
            
            # Destroy the panel and root
            panel.destroy()
            root.destroy()
            
            print("\nUI test completed (simplified)")
            return True
            
        except Exception as e:
            print(f"❌ UI test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        test_results = {
            "bridge": await self.test_bridge(),
            "visualizer": await self.test_visualizer(),
            "settings": await self.test_settings(),
            "deployment": await self.test_deployment(skip_full_deployment=True),
            "ui": await self.test_ui()
        }
        
        # Print summary
        print("\n=========================")
        print("📊 Test Results Summary")
        print("=========================")
        
        for test_name, result in test_results.items():
            print(f"{test_name.ljust(15)}: {'✅ Passed' if result else '❌ Failed'}")
        
        success_rate = sum(1 for r in test_results.values() if r) / len(test_results) * 100
        print(f"\nSuccess rate: {success_rate:.1f}%")
        
        return all(test_results.values())


async def main():
    """Main function"""
    test = QuantumIntegrationTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
