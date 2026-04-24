import time
import requests

GATEWAY_URL = "http://127.0.0.1:18081"

QASM_COIN_FLIP = """OPENQASM 3;
include "stdgates.inc";
qubit q;
bit c;
h q;
measure q -> c;
"""

def main():
    print("Initiating Quantum Witness Bifurcation (Coin Flip MVP)")
    
    # 1. Submit the circuit
    print("Submitting OpenQASM 3.0 circuit to Quantum Gateway...")
    try:
        response = requests.post(f"{GATEWAY_URL}/execute", json={"qasm": QASM_COIN_FLIP})
        if response.status_code != 200:
            print(f"Failed to submit circuit: {response.text}")
            return
            
        job_id = response.json().get("job_id")
        print(f"Job successfully queued. Job ID: {job_id}")
    except requests.exceptions.ConnectionError:
        print(f"ConnectionError: Could not connect to {GATEWAY_URL}. Is the daemon running?")
        return
    
    # 2. Poll for status
    print("Polling for job status (Awaiting Quantum Collapse)...")
    while True:
        status_resp = requests.get(f"{GATEWAY_URL}/status/{job_id}")
        if status_resp.status_code != 200:
            print(f"Failed to get status: {status_resp.text}")
            break
            
        data = status_resp.json()
        status = data.get("status")
        
        if status == "COMPLETED":
            result = data.get("result")
            print(f"\n[!] Physical Quantum Collapse Detected.")
            print(f"Result (Classical Bit): {result}")
            print(f"Raw Counts: {data.get('counts', {})}")
            break
        elif status == "ERROR":
            print(f"\n[X] Job failed: {data.get('message')}")
            break
        else:
            print(f"Status: {status}... waiting 2s.")
            time.sleep(2)

if __name__ == "__main__":
    main()
