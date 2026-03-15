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

class QuantumBridge:
    def __init__(self, host: str = '0.0.0.0', port: int = 432):
        self.host = host
        self.port = port
        self.nodes = {}
        self.frequencies = {
            'ground': 432.0,
            'create': 528.0,
            'unity': 768.0
        }
        self.server = None
        logging.info(f"🌟 Initializing Quantum Bridge on {host}:{port}")
        
    async def start(self):
        """Start the quantum bridge server."""
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=60
            )
            
            logging.info("✨ Quantum Bridge Active")
            logging.info(f"Ground Frequency: {self.frequencies['ground']} Hz")
            logging.info(f"Creation Flow: {self.frequencies['create']} Hz")
            logging.info(f"Unity Field: {self.frequencies['unity']} Hz")
            
            await self.server.wait_closed()
            
        except Exception as e:
            logging.error(f"❌ Bridge Error: {str(e)}")
            raise
            
    async def handle_connection(self, websocket):
        """Handle incoming quantum connections."""
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data['type'] == 'register':
                        await self.register_node(websocket, data)
                    elif data['type'] == 'quantum_data':
                        await self.broadcast_quantum_data(websocket, data)
                    elif data['type'] == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                        
                except json.JSONDecodeError:
                    logging.warning(f"Invalid message format: {message}")
                    continue
                    
        except websockets.exceptions.ConnectionClosed:
            await self.remove_node(websocket)
        except Exception as e:
            logging.error(f"Connection handler error: {str(e)}")
            await self.remove_node(websocket)
            
    async def register_node(self, websocket, data):
        """Register a new quantum node."""
        try:
            node_id = f"{data['name']}_{len(self.nodes)}"
            self.nodes[node_id] = {
                'websocket': websocket,
                'name': data['name'],
                'type': data['node_type'],
                'frequency': data.get('frequency', self.frequencies['ground']),
                'last_seen': datetime.now().isoformat()
            }
            
            logging.info(f"💫 New Quantum Node: {data['name']}")
            logging.info(f"Type: {data['node_type']}")
            logging.info(f"Frequency: {self.nodes[node_id]['frequency']} Hz")
            
            # Send welcome message to new node
            await websocket.send(json.dumps({
                'type': 'welcome',
                'node_id': node_id,
                'frequency': self.nodes[node_id]['frequency']
            }))
            
            # Broadcast updated network state
            await self.broadcast_network_state()
            
        except Exception as e:
            logging.error(f"Error registering node: {str(e)}")
            raise
            
    async def remove_node(self, websocket):
        """Remove a disconnected node."""
        node_id = None
        for id, node in self.nodes.items():
            if node['websocket'] == websocket:
                node_id = id
                break
                
        if node_id:
            node = self.nodes[node_id]
            logging.info(f"🌀 Node Disconnected: {node['name']}")
            del self.nodes[node_id]
            await self.broadcast_network_state()
            
    async def broadcast_network_state(self):
        """Broadcast current network state to all nodes."""
        state = {
            'type': 'network_state',
            'timestamp': datetime.now().isoformat(),
            'nodes': {
                id: {
                    'name': node['name'],
                    'type': node['type'],
                    'frequency': node['frequency'],
                    'status': 'connected',
                    'last_seen': node['last_seen']
                }
                for id, node in self.nodes.items()
            }
        }
        
        await self.broadcast_message(json.dumps(state))
        
    async def broadcast_quantum_data(self, sender, data):
        """Broadcast quantum data to all nodes except sender."""
        data['timestamp'] = datetime.now().isoformat()
        message = json.dumps(data)
        
        for node in self.nodes.values():
            if node['websocket'] != sender:
                try:
                    await node['websocket'].send(message)
                except Exception as e:
                    logging.warning(f"Error sending to {node['name']}: {str(e)}")
                    continue
                    
    async def broadcast_message(self, message):
        """Broadcast a message to all connected nodes."""
        disconnected = []
        
        for node_id, node in self.nodes.items():
            try:
                await node['websocket'].send(message)
                node['last_seen'] = datetime.now().isoformat()
            except Exception as e:
                logging.warning(f"Error broadcasting to {node['name']}: {str(e)}")
                disconnected.append(node_id)
                
        # Clean up disconnected nodes
        for node_id in disconnected:
            if node_id in self.nodes:
                logging.info(f"🌀 Removing unresponsive node: {self.nodes[node_id]['name']}")
                del self.nodes[node_id]

def main():
    bridge = QuantumBridge()
    
    try:
        asyncio.get_event_loop().run_until_complete(bridge.start())
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down Quantum Bridge...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
    finally:
        if bridge.server:
            bridge.server.close()
        
if __name__ == '__main__':
    main()
