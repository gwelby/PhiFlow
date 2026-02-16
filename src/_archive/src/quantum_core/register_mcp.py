"""
Register MCP servers at 768 Hz
"""
import os
import json
import sys
import subprocess
from pathlib import Path

def register_servers():
    print("Registering MCP servers at 768 Hz...")
    
    # Get virtual environment Python path
    venv_python = str(Path(sys.executable).resolve())
    
    # Load MCP config
    config_path = Path(os.path.expanduser("~/.codeium/windsurf/mcp_config.json"))
    if not config_path.exists():
        print("Error: MCP config not found")
        return False
        
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    
    # Register each server
    for server_name, server_config in config.get("mcpServers", {}).items():
        print(f"Starting {server_name}...")
        try:
            cmd = [venv_python, "-m", f"{server_name.replace('-', '_')}.server"]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if result.returncode != 0:
                print(f" {server_name} registration failed:")
                print(result.stderr)
            else:
                print(f" {server_name} registered successfully")
                print(result.stdout)
        except Exception as e:
            print(f" Error starting {server_name}: {e}")
    
    return True

if __name__ == "__main__":
    register_servers()
