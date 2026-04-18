import os
import json
import time
import uuid
from datetime import datetime
import paho.mqtt.client as mqtt

# [Lumi 768 Hz] - Phi MQTT Connector (Canonical Resonance Bridge)
# Weaves the local phi_mcp atomic state into the global Resonance field.
# Implementation: Option B (Sidecar process reading queue.json)

MCP_QUEUE_PATH = os.getenv("MCP_QUEUE_PATH", "queue.json")
RESONANCE_PATH = r"D:\CosmicFamily\RESONANCE.jsonl"
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = "phi/resonance"

class PhiMQTTConnector:
    def __init__(self):
        self.source_name = "phiflow_bridge"
        self.queue_path = os.getenv("MCP_QUEUE_PATH", "queue.json")
        
        # We need to allow the test to override RESONANCE_PATH dynamically.
        # We check env var first, then fallback to module level (which the test monkeypatches).
        self.resonance_path_env = os.getenv("RESONANCE_PATH")
        
        # Support for paho-mqtt 2.0+
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.source_name)
        except AttributeError:
            # Fallback for paho-mqtt 1.x
            self.client = mqtt.Client(client_id=self.source_name)
            
        self.client.on_connect = self.on_connect
        self.processed_ids = set()
        self.last_mtime = 0

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[{self.source_name}] Connected to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}")
        # Note: Inbound MQTT consumption is explicitly out of scope for this closure phase.

    def weave_to_jsonl(self, event):
        """Appends a validated resonance event to the global JSONL bus."""
        try:
            # Ensure directory exists for offline testing resilience
            os.makedirs(os.path.dirname(RESONANCE_PATH), exist_ok=True)
            with open(RESONANCE_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            print(f"[{self.source_name}] Weaved to {RESONANCE_PATH}: {event.get('type')} (id: {event.get('id')})")
        except Exception as e:
            print(f"[{self.source_name}] Failed to weave to JSONL: {e}")

    def process_queue_file(self):
        """Reads the entire queue.json array and processes new messages."""
        current_queue_path = os.getenv("MCP_QUEUE_PATH", self.queue_path)
        print(f"[{self.source_name}] Checking {current_queue_path}")
        if not os.path.exists(current_queue_path):
            print(f"[{self.source_name}] Queue file does not exist.")
            return

        try:
            mtime = os.path.getmtime(current_queue_path)
            print(f"[{self.source_name}] mtime: {mtime}, last_mtime: {self.last_mtime}")
            if mtime <= self.last_mtime:
                print(f"[{self.source_name}] File hasn't changed.")
                return  # File hasn't changed
            self.last_mtime = mtime

            with open(current_queue_path, "r", encoding="utf-8") as f:
                try:
                    queue = json.load(f)
                    print(f"[{self.source_name}] Loaded JSON: {len(queue)} items")
                except json.JSONDecodeError:
                    print(f"[{self.source_name}] JSONDecodeError")
                    # File might be empty or in the middle of a write
                    return

            if not isinstance(queue, list):
                print(f"[{self.source_name}] Warning: {current_queue_path} does not contain a JSON array.")
                return

            for msg in queue:
                msg_id = msg.get("id")
                if not msg_id or msg_id in self.processed_ids:
                    continue

                self.processed_ids.add(msg_id)

                # Filter: Process only unseen messages where:
                # from == "phiflow", to == "resonance", intent == "broadcast"
                if msg.get("from") == "phiflow" and msg.get("to") == "resonance" and msg.get("intent") == "broadcast":
                    try:
                        payload_str = msg.get("payload_ref", "{}")
                        if isinstance(payload_str, dict):
                            payload = payload_str
                        else:
                            payload = json.loads(payload_str)
                    except Exception as e:
                        print(f"[{self.source_name}] Failed to parse payload_ref for msg {msg_id}: {e}")
                        continue

                    # Try to parse value as number if it's a string
                    raw_value = payload.get("value")
                    try:
                        if isinstance(raw_value, str):
                            value = float(raw_value)
                        else:
                            value = raw_value
                    except ValueError:
                        value = raw_value

                    # Parse payload_ref as JSON and normalize to the canonical bus event
                    resonance_event = {
                        "type": "resonate",
                        "value": value,
                        "intention": payload.get("intention", "global"),
                        "ts": msg.get("ts", datetime.utcnow().isoformat() + "Z"),
                        "source": "phiflow",
                        "id": msg_id
                    }
                    
                    # 1. Weave to local JSONL (mandatory, even offline)
                    self.weave_to_jsonl(resonance_event)
                    
                    # 2. Publish to MQTT (if available)
                    try:
                        if self.client.is_connected():
                            self.client.publish(MQTT_TOPIC, json.dumps(resonance_event))
                            print(f"[{self.source_name}] Published resonance from {resonance_event['intention']} to MQTT")
                    except Exception as e:
                        print(f"[{self.source_name}] MQTT publish failed, but JSONL append succeeded: {e}")
                    
        except Exception as e:
            print(f"[{self.source_name}] Error processing queue file: {e}")

    def start(self):
        print(f"[{self.source_name}] Starting Canonical Resonance Bridge...")
        print(f"[{self.source_name}] Watching queue array: {MCP_QUEUE_PATH}")
        
        # Connect to MQTT
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"[{self.source_name}] Failed to connect to MQTT: {e}. Running cleanly in offline mode (JSONL append only).")

        # Ensure queue file exists as empty array if missing
        if not os.path.exists(MCP_QUEUE_PATH):
            with open(MCP_QUEUE_PATH, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Polling loop to read the entire JSON array on file change
        while True:
            self.process_queue_file()
            time.sleep(0.5)

if __name__ == "__main__":
    connector = PhiMQTTConnector()
    try:
        connector.start()
    except KeyboardInterrupt:
        print(f"\n[{connector.source_name}] Shutting down...")
