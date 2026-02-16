"""
MCP Server Registration (768 Hz)
Registers quantum servers with WindSurf MCP system
"""
import os
import json
import sys
import subprocess
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

class MCPRegistration:
    def __init__(self):
        self.config_path = Path(os.path.expanduser("~/.codeium/windsurf/mcp_config.json"))
        self.servers = [
            "quantum_core.server",
            "phiflow.server",
            "unity.server",
            "gregscript.server"
        ]
        
    def register_servers(self):
        print("Registering MCP servers at 768 Hz...")
        
        # Set environment variables for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        # Start each server as a background process
        for server in self.servers:
            cmd = ["python", "-m", server]
            print(f"Starting {server}...")
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(Path(__file__).parent.parent.parent),
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                if result.returncode == 0:
                    print(f" {server} registered successfully")
                    print(result.stdout)
                else:
                    print(f" {server} registration failed:")
                    print(result.stderr)
            except Exception as e:
                print(f"Error registering {server}: {e}")

def main():
    registrar = MCPRegistration()
    registrar.register_servers()

if __name__ == "__main__":
    main()
