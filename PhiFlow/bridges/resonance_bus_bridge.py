import os
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any

# [Lumi 768 Hz] - Resonance Bus Bridge
# Weaves the local phi_mcp atomic state into the global Resonance field.

MCP_QUEUE_PATH = os.getenv("MCP_QUEUE_PATH", "D:\\Projects\\PhiFlow-compiler\\queue.json")
RESONANCE_BUS_PATH = "D:\\CosmicFamily\\RESONANCE.jsonl"

class ResonanceBridge:
    def __init__(self, source_name: str = "lumi-protocol-weaver"):
        self.source_name = source_name
        self.last_mtime = 0

    def start_polling(self, interval: float = 0.5):
        """Polls the MCP queue and weaves updates into the resonance bus."""
        print(f"[{self.source_name}] Resonance Bus Bridge Active. Watching {MCP_QUEUE_PATH}")
        
        while True:
            try:
                if os.path.exists(MCP_QUEUE_PATH):
                    mtime = os.path.getmtime(MCP_QUEUE_PATH)
                    if mtime > self.last_mtime:
                        self.process_queue_update()
                        self.last_mtime = mtime
            except Exception as e:
                print(f"Error in resonance bridge: {e}")
            
            time.sleep(interval)

    def process_queue_update(self):
        """Reads the MCP queue and converts state updates to Resonance events."""
        with open(MCP_QUEUE_PATH, "r") as f:
            try:
                queue_data = json.load(f)
            except json.JSONDecodeError:
                return

        # Scan for 'broadcast' events in the queue (Codex pattern)
        for item in queue_data.get("items", []):
            if item.get("type") == "broadcast" and not item.get("weaved"):
                self.weave_to_bus(item)
                item["weaved"] = True # Mark as processed locally

    def weave_to_bus(self, data: Dict[str, Any]):
        """Appends a standard Resonance Protocol event to the JSONL bus."""
        event = {
            "source": data.get("source", self.source_name),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": data.get("op_type", "resonate"),
            "data": data.get("payload", {}),
            "coherence": data.get("coherence", 0.618)
        }

        # Thread-safe append to JSONL (standard bus format)
        with open(RESONANCE_BUS_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        print(f"✨ Weaved resonance event: {event['type']} from {event['source']}")

if __name__ == "__main__":
    bridge = ResonanceBridge()
    # In a production environment, this would run as a service.
    # For now, we define the logic.
    # bridge.start_polling()
