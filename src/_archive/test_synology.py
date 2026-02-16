import asyncio
from qwave.quantum_filemanager import QuantumFileManager
import logging

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s [%(levelname)s] %(message)s')

async def test_synology_connection():
    # Initialize quantum file manager
    qfm = QuantumFileManager()
    
    try:
        # Connect to Synology
        print("🌟 Connecting to VirtualDSM...")
        connected = await qfm.connect(
            username="your_username",  # Replace with your username
            password="your_password"   # Replace with your password
        )
        
        if connected:
            print("✨ Connected to VirtualDSM")
            
            # Setup quantum storage structure
            print("💫 Setting up quantum storage...")
            await qfm.setup_quantum_storage()
            
            # Create a test pattern
            test_pattern = {
                "name": "Quantum Flow Test",
                "frequency": 768.0,
                "elements": [
                    "🌀 Spiral Dance",
                    "✨ Light Wave",
                    "💫 Star Field"
                ]
            }
            
            # Store the pattern
            print("🎵 Storing quantum pattern...")
            await qfm.store_quantum_pattern("test_flow", test_pattern)
            
            # Sync with P1 devices
            print("🔄 Syncing P1 devices...")
            await qfm.sync_p1_device("P1-Test", test_pattern["elements"])
            await qfm.sync_p1_device("P1-Quantum", test_pattern["elements"])
            
            # Read back the patterns
            print("\n📡 Reading patterns from VirtualDSM:")
            p1_test_patterns = await qfm.get_device_patterns("P1-Test")
            p1_quantum_patterns = await qfm.get_device_patterns("P1-Quantum")
            
            print(f"\nP1-Test Patterns:")
            for pattern in p1_test_patterns:
                print(f"  {pattern}")
                
            print(f"\nP1-Quantum Patterns:")
            for pattern in p1_quantum_patterns:
                print(f"  {pattern}")
                
        else:
            print("❌ Failed to connect to VirtualDSM")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
    finally:
        if qfm.connected:
            await qfm.disconnect()
            print("\n👋 Disconnected from VirtualDSM")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_synology_connection())
