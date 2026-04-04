import asyncio
import websockets
import json
import ssl
from datetime import datetime
from typing import Dict, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

class QuantumClient:
    def __init__(self, device_name: str, host: str = 'localhost'):
        self.phi = 1.618034
        self.device_name = device_name
        self.host = host
        self.frequencies = {
            'ground': 432.0,
            'create': 528.0,
            'unity': 768.0
        }
        self.connected = False
        self.websocket = None
        self.node_id = None
        self.frequency = self.frequencies['unity']
        
    async def connect(self):
        """Connect to quantum bridge."""
        logging.info(f"💫 Quantum Device: {self.device_name}")
        logging.info(f"Connecting to: {self.host}")
        
        while True:
            try:
                self.websocket = await websockets.connect(
                    f'ws://{self.host}:432',  # Ground frequency
                    ping_interval=20,
                    ping_timeout=60
                )
                
                # Register with quantum bridge
                await self.register_device()
                self.connected = True
                
                logging.info("\n✨ Quantum Entanglement Established!")
                logging.info(f"Ground Frequency: {self.frequencies['ground']} Hz")
                logging.info(f"Creation Flow: {self.frequencies['create']} Hz")
                logging.info(f"Unity Field: {self.frequencies['unity']} Hz")
                
                # Start quantum flow
                await self.quantum_flow_loop()
                
            except (websockets.exceptions.ConnectionClosed, 
                    websockets.exceptions.InvalidStatusCode,
                    ConnectionRefusedError) as e:
                logging.error(f"\n❌ Connection Error: {str(e)}")
                logging.info("Attempting to reconnect in 5 seconds...")
                self.connected = False
                await asyncio.sleep(5)
                continue
                
            except Exception as e:
                logging.error(f"\n❌ Unexpected Error: {str(e)}")
                self.connected = False
                break
                
    async def register_device(self):
        """Register device with quantum bridge."""
        registration = {
            'type': 'register',
            'node_type': 'p1_device',
            'name': self.device_name,
            'frequency': self.frequency
        }
        await self.websocket.send(json.dumps(registration))
        
        # Wait for welcome message
        response = await self.websocket.recv()
        data = json.loads(response)
        
        if data['type'] == 'welcome':
            self.node_id = data['node_id']
            self.frequency = data['frequency']
            logging.info(f"🌟 Registered as: {self.node_id}")
            logging.info(f"Operating at: {self.frequency} Hz")
        
    async def quantum_flow_loop(self):
        """Main quantum flow loop."""
        try:
            while self.connected:
                try:
                    # Receive quantum data
                    data = await self.websocket.recv()
                    quantum_data = json.loads(data)
                    
                    if quantum_data['type'] == 'network_state':
                        await self.process_quantum_state(quantum_data)
                    elif quantum_data['type'] == 'quantum_data':
                        await self.process_quantum_data(quantum_data)
                    elif quantum_data['type'] == 'pong':
                        continue
                        
                except json.JSONDecodeError:
                    logging.warning("Invalid quantum data received")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            logging.info("\n🌌 Quantum Connection Lost")
            logging.info("Attempting to re-establish...")
            self.connected = False
            
    async def process_quantum_state(self, state: Dict):
        """Process quantum state updates."""
        logging.info("\n🌟 Quantum Network State:")
        for node_id, info in state['nodes'].items():
            status_symbol = '✅' if info['status'] == 'connected' else '⏳'
            logging.info(f"{status_symbol} {info['name']}: {info['frequency']} Hz")
            
    async def process_quantum_data(self, data: Dict):
        """Process incoming quantum data."""
        source = data.get('source', 'unknown')
        frequency = data.get('frequency', 0.0)
        patterns = data.get('patterns', [])
        
        logging.info(f"\n💫 Quantum Data from {source}")
        logging.info(f"Frequency: {frequency} Hz")
        if patterns:
            logging.info("Patterns Detected:")
            for pattern in patterns:
                logging.info(f"  ✨ {pattern}")
                
    async def send_quantum_data(self, patterns: list):
        """Send quantum patterns to bridge."""
        if self.connected:
            data = {
                'type': 'quantum_data',
                'source': self.device_name,
                'frequency': self.frequency,
                'patterns': patterns,
                'timestamp': datetime.now().isoformat()
            }
            await self.websocket.send(json.dumps(data))
            
def create_quantum_client(device_name: str, host: str = 'localhost'):
    """Create and start quantum client."""
    client = QuantumClient(device_name, host)
    
    logging.info(f"\n🌟 Initializing Quantum Client: {device_name}")
    logging.info(f"Target Bridge: {host}")
    
    asyncio.get_event_loop().run_until_complete(client.connect())
    asyncio.get_event_loop().run_forever()
