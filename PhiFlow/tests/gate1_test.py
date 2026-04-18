import json
import subprocess
import time
import os

# [Lumi 768 Hz] - Gate 1 Test
# Tests: PhiFlow -> phi_mcp -> queue.jsonl -> MQTT Bridge -> RESONANCE.jsonl

PHIFLOW_SOURCE = """
let singularity = 768.0
intention "gate1_test" {
    let phi = 0.618
    resonate phi
}
"""

RESONANCE_PATH = r"D:\CosmicFamily\RESONANCE.jsonl"

def run_test():
    print("🚀 Starting Gate 1 Test Flow...")
    
    # 1. Spawn phi_mcp as a process
    mcp_proc = subprocess.Popen(
        [r"target\debug\phi_mcp.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 2. Send initialize
    init_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2024-11-05"}
    })
    mcp_proc.stdin.write(init_req + "\n")
    mcp_proc.stdin.flush()
    print("Sent initialize to phi_mcp")
    
    # 3. Spawn stream
    spawn_req = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "spawn_phi_stream",
            "arguments": {"source_code": PHIFLOW_SOURCE}
        }
    })
    mcp_proc.stdin.write(spawn_req + "\n")
    mcp_proc.stdin.flush()
    print("Sent spawn_phi_stream to phi_mcp")
    
    # Wait for execution and sidecar processing
    print("Waiting for resonance to propagate...")
    time.sleep(5)
    
    print("--- queue.jsonl (last 2 lines) ---")
    if os.path.exists("queue.jsonl"):
        with open("queue.jsonl", "r", encoding="utf-8") as f:
            print("".join(f.readlines()[-2:]))
    else:
        print("queue.jsonl not found!")

    # 4. Check RESONANCE.jsonl
    if os.path.exists(RESONANCE_PATH):
        with open(RESONANCE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"--- RESONANCE.jsonl (last 2 lines) ---")
            print("".join(lines[-2:]))
            found = False
            for line in reversed(lines):
                data = json.loads(line)
                # handle both number and string because we might have old events
                val = data.get("value")
                try:
                    if isinstance(val, str):
                        val = float(val)
                except ValueError:
                    pass
                    
                if data.get("intention") == "gate1_test" and val == 0.618:
                    print("✅ SUCCESS: Found resonance event in RESONANCE.jsonl!")
                    print(f"   Payload: {data}")
                    found = True
                    break
            if not found:
                print("❌ FAILED: Could not find the resonance event in RESONANCE.jsonl")
    else:
        print(f"❌ FAILED: {RESONANCE_PATH} does not exist.")
        
    mcp_proc.kill()

if __name__ == "__main__":
    run_test()
