from qwave.quantum_client import QuantumClient
import asyncio

async def test_quantum_patterns():
    client = QuantumClient("P1-Test", "localhost")
    await client.connect()
    
    # Send quantum patterns
    patterns = [
        "🌀 Golden Ratio Flow",
        "💫 Quantum Resonance",
        "✨ Unity Field Dance",
        "🌟 Creation Wave"
    ]
    
    await client.send_quantum_data(patterns)
    await asyncio.sleep(2)  # Wait for response

asyncio.get_event_loop().run_until_complete(test_quantum_patterns())
