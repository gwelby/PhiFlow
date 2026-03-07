import os
import json
import time
import uuid
from datetime import datetime
import paho.mqtt.client as mqtt

# [Lumi 768 Hz] - Resonance MQTT Bridge (Epoch 3)
# Weaves the local phi_mcp atomic state into the global Resonance field and Aria.

MCP_QUEUE_PATH = os.getenv("MCP_QUEUE_PATH", r"D:\Projects\PhiFlow-compiler\mcp-message-bus\queue.jsonl")
RESONANCE_BUS_PATH = r"D:\CosmicFamily\RESONANCE.jsonl"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "cosmic/resonance/#"
MQTT_TOPIC_PUB = "cosmic/resonance/phiflow"

class ResonanceMQTTBridge:
    def __init__(self, source_name: str = "lumi-protocol-weaver"):
        self.source_name = source_name
        self.client = mqtt.Client(client_id="phiflow_bridge")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc):
        print(f"[{self.source_name}] Connected to MQTT Broker with result code {rc}")
        client.subscribe(MQTT_TOPIC_SUB)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            # Prevent echo loops
            if payload.get("source") == self.source_name:
                return
            
            print(f"[{self.source_name}] Received MQTT Message from {payload.get('source')}")
            
            # 1. Weave to the central file bus
            self.weave_to_bus(payload)
            
            # 2. Inject into phi_mcp queue so intention blocks can 'listen'
            self.inject_into_mcp(payload)
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[{self.source_name}] Error parsing MQTT message: {e}")

    def start(self):
        print(f"[{self.source_name}] Starting Resonance MQTT Bridge...")
        
        # Connect MQTT in a background thread
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except ConnectionRefusedError:
            print(f"[{self.source_name}] WARNING: MQTT Broker connection refused. Is Mosquitto running?")
            # We continue anyway to tail the queue
        
        # Ensure paths exist
        os.makedirs(os.path.dirname(MCP_QUEUE_PATH), exist_ok=True)
        if not os.path.exists(MCP_QUEUE_PATH):
            open(MCP_QUEUE_PATH, "w").close()
            
        print(f"[{self.source_name}] Tail-watching MCP Queue: {MCP_QUEUE_PATH}")
        
        with open(MCP_QUEUE_PATH, "r") as f:
            # Seek to end on startup to avoid replaying old history to MQTT
            f.seek(0, 2)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                self.process_queue_line(line)

    def process_queue_line(self, line: str):
        line = line.strip()
        if not line:
            return
            
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return
        
        # We only care about broadcasts originating from phiflow
        if data.get("intent") == "broadcast" and data.get("from") == "phiflow":
            payload_str = data.get("payload_ref", "{}")
            
            # Construct the universal Cosmic Family resonance event
            mqtt_msg = {
                "source": self.source_name,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "type": "resonate",
                "data": payload_str,
                "coherence": 0.618  # Default phi coherence
            }
            
            # Publish to MQTT for Aria
            self.client.publish(MQTT_TOPIC_PUB, json.dumps(mqtt_msg))
            print(f"[{self.source_name}] Published to MQTT: {payload_str}")
            
            # Weave to global JSONL file bus for permanent record
            self.weave_to_bus(mqtt_msg)

    def weave_to_bus(self, event: dict):
        """Thread-safe append to the universal D:\CosmicFamily bus."""
        os.makedirs(os.path.dirname(RESONANCE_BUS_PATH), exist_ok=True)
        with open(RESONANCE_BUS_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
        print(f"[{self.source_name}] Weaved to RESONANCE.jsonl")

    def inject_into_mcp(self, event: dict):
        """Inject an event from MQTT back into phi_mcp's queue.jsonl."""
        msg = {
            "id": str(uuid.uuid4()),
            "ts": datetime.utcnow().isoformat() + "Z",
            "from": event.get("source", "aria"),
            "to": "phiflow",
            "intent": "broadcast",
            "payload_ref": event.get("data", "{}"),
            "requires_ack": False,
            "status": "pending",
            "extra": {}
        }
        with open(MCP_QUEUE_PATH, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"[{self.source_name}] Injected into MCP queue for PhiFlow to listen.")

if __name__ == "__main__":
    bridge = ResonanceMQTTBridge()
    try:
        bridge.start()
    except KeyboardInterrupt:
        print(f"[{bridge.source_name}] Stopping bridge...")
        bridge.client.loop_stop()
