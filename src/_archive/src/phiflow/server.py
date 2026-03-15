"""
PhiFlow Server (528 Hz)
Pattern creation and flow states
"""
import os
import json
import sys
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class PhiFlowServer:
    def __init__(self):
        self.frequency = 528  # DNA/Creation frequency
        self.phi_squared = 2.618034  # φ²
        self.config = self._load_config()
        
    def _load_config(self):
        config_path = Path(os.path.expanduser("~/.codeium/windsurf/mcp_config.json"))
        if config_path.exists():
            with open(config_path, encoding='utf-8') as f:
                return json.load(f)
        return {}
        
    def start(self):
        print(f"Starting PhiFlow Server at {self.frequency} Hz")
        tools = self.config.get("quantumSettings", {}).get("tools", {}).get("flow", [])
        patterns = self.config.get("quantumSettings", {}).get("patterns", {})
        
        print("Flow tools enabled:", ", ".join(str(t) for t in tools))
        print("Sacred patterns:", " ".join(f"{k} ({v})" for k, v in patterns.items()))
        print(f"Phi squared resonance: {self.phi_squared}")
        return True

def main():
    server = PhiFlowServer()
    server.start()

if __name__ == "__main__":
    main()
