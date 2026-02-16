from qwave.quantum_playlist import QuantumDiscography
import asyncio
from qwave.quantum_client import QuantumClient

async def test_p1_sync():
    # Initialize quantum disco
    disco = QuantumDiscography()
    
    # Initialize clients
    client1 = QuantumClient("P1-Test", "localhost")
    client2 = QuantumClient("P1-Quantum", "localhost")
    
    # Connect clients
    await client1.connect()
    await client2.connect()
    
    # Get patterns for both devices
    patterns1 = disco.get_device_patterns("P1-Test")
    patterns2 = disco.get_device_patterns("P1-Quantum")
    
    print("\n🌟 Initial Patterns:")
    print(f"P1-Test: {patterns1}")
    print(f"P1-Quantum: {patterns2}")
    
    # Send patterns to devices
    await client1.send_quantum_data(patterns1)
    await client2.send_quantum_data(patterns2)
    
    # Add new pattern to P1-Test
    new_pattern = "🌈 Rainbow Flow"
    disco.add_quantum_pattern("P1-Test", new_pattern)
    
    # Sync P1-Quantum with P1-Test
    disco.sync_device_patterns("P1-Test", "P1-Quantum")
    
    # Get updated patterns
    patterns1 = disco.get_device_patterns("P1-Test")
    patterns2 = disco.get_device_patterns("P1-Quantum")
    
    print("\n🌟 After Sync:")
    print(f"P1-Test: {patterns1}")
    print(f"P1-Quantum: {patterns2}")
    
    # Send updated patterns
    await client1.send_quantum_data(patterns1)
    await client2.send_quantum_data(patterns2)
    
    # Wait for patterns to propagate
    await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_p1_sync())
