import os
import json
import time
import random
import threading
import uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Stage 2: Quantum Witness Bridge
# Target: ibm_heron (ibm_fez)
# Runs a background thread for presence polling AND an HTTP server for execution targeting.

PORT = 18081
MOCK_MODE = os.getenv("PHIFLOW_MOCK_QUANTUM") == "1" or not os.getenv("IBMQ_API_TOKEN")
STATE_PATH = os.getenv("PHIFLOW_QUANTUM_STATE_PATH", "D:/Projects/PhiHarmonic/SOMA/quantum_state.json")
BACKEND_NAME = os.getenv("IBMQ_BACKEND", "ibm_fez")

# Lazy Qiskit Service
_service = None

def get_service():
    global _service
    if _service is None and not MOCK_MODE:
        from qiskit_ibm_runtime import QiskitRuntimeService
        token = os.getenv("IBMQ_API_TOKEN")
        _service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    return _service

# In-memory mock job registry
_mock_jobs = {}
_mock_job_lock = threading.Lock()

def get_quantum_metrics():
    if MOCK_MODE:
        return {
            "status": "mock",
            "backend": f"{BACKEND_NAME}_mock",
            "updated_at": datetime.now().isoformat(),
            "metrics": {
                "quantum_t1": round(random.uniform(150.0, 250.0), 2),
                "quantum_t2": round(random.uniform(80.0, 180.0), 2),
                "quantum_readout_error": round(random.uniform(0.005, 0.02), 4)
            }
        }

    try:
        service = get_service()
        backend = service.backend(BACKEND_NAME)
        props = backend.properties()
        
        t1s = [props.t1(i) * 1e6 for i in range(backend.num_qubits)]
        t2s = [props.t2(i) * 1e6 for i in range(backend.num_qubits)]
        errors = [props.readout_error(i) for i in range(backend.num_qubits)]

        return {
            "status": "live",
            "backend": BACKEND_NAME,
            "updated_at": datetime.now().isoformat(),
            "metrics": {
                "quantum_t1": sum(t1s) / len(t1s),
                "quantum_t2": sum(t2s) / len(t2s),
                "quantum_readout_error": sum(errors) / len(errors)
            }
        }
    except Exception as e:
        return {"error": str(e)}

def telemetry_loop():
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    print("Quantum Telemetry thread started.")
    
    while True:
        result = get_quantum_metrics()
        try:
            with open(STATE_PATH, "w") as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"Failed to write telemetry: {e}")
            
        time.sleep(5 if MOCK_MODE else 60)

def execute_circuit_live(qasm_string):
    from qiskit import qasm3
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2 as Sampler
    
    service = get_service()
    backend = service.backend(BACKEND_NAME)
    
    # Parse OpenQASM 3.0 string (qasm3.loads is the correct Qiskit 1.x API)
    qc = qasm3.loads(qasm_string)
    
    # Transpile to backend ISA — required by IBM hardware before submission
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qc)
    
    # Execute via SamplerV2 using mode= constructor (not backend=)
    sampler = Sampler(mode=backend)
    job = sampler.run([isa_circuit])
    
    print(f"Live Job Submitted: {job.job_id()} on {BACKEND_NAME}")
    return job.job_id()

def get_job_status_live(job_id):
    service = get_service()
    job = service.job(job_id)
    # In qiskit-ibm-runtime >= 0.21, job.status() returns a plain string:
    # 'INITIALIZING', 'QUEUED', 'RUNNING', 'DONE', 'ERROR', 'CANCELLED'
    status = job.status()
    
    if status == "DONE":
        result = job.result()
        # V2 Sampler returns PubResult list; extract first pub
        pub_result = result[0]
        data = pub_result.data
        
        # Find the classical register name dynamically.
        # IBM transpiler typically names the output register 'meas'.
        # Fallback: check for 'c', then any first register found.
        counts = {}
        for reg_name in ('meas', 'c'):
            if hasattr(data, reg_name):
                counts = getattr(data, reg_name).get_counts()
                break
        else:
            keys = list(data.keys())
            if keys:
                counts = getattr(data, keys[0]).get_counts()
            
        # Collapse the multi-shot result into a single dominant bit string
        if not counts:
            collapsed_bit = "0"
        else:
            collapsed_bit = max(counts.items(), key=lambda x: x[1])[0]

        return {"status": "COMPLETED", "result": str(collapsed_bit), "counts": counts}
    elif status in ("INITIALIZING", "QUEUED"):
        return {"status": "QUEUED", "result": None}
    elif status == "RUNNING":
        return {"status": "RUNNING", "result": None}
    else:
        return {"status": "ERROR", "message": str(status)}

def execute_circuit_mock(qasm_string):
    job_id = str(uuid.uuid4())
    print(f"Mocking QASM Execution. Job ID: {job_id}")
    with _mock_job_lock:
        _mock_jobs[job_id] = {
            "status": "QUEUED",
            "submit_time": time.time(),
            "qasm": qasm_string,
            "result": None
        }
    return job_id

def get_job_status_mock(job_id):
    with _mock_job_lock:
        job = _mock_jobs.get(job_id)
        if not job:
            return {"status": "ERROR", "message": "Job not found."}
        
        # Simulate queue time (2 seconds)
        if time.time() - job["submit_time"] > 2.0:
            if job["status"] != "COMPLETED":
                job["status"] = "COMPLETED"
                # Random bit for mock
                bit = random.choice(["0", "1"])
                job["result"] = bit
                print(f"Mock Job {job_id} Completed. Result: {job['result']}")
                
        return {"status": job["status"], "result": job["result"]}

class QuantumGatewayHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/execute":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode('utf-8'))
                qasm = payload.get("qasm")
                if not qasm:
                    self.send_error(400, "Missing 'qasm' in payload")
                    return
                
                if MOCK_MODE:
                    job_id = execute_circuit_mock(qasm)
                else:
                    job_id = execute_circuit_live(qasm)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"job_id": job_id}).encode('utf-8'))
            except Exception as e:
                print(f"Error executing circuit: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/status/"):
            job_id = parsed.path.split("/")[-1]
            try:
                if MOCK_MODE:
                    status = get_job_status_mock(job_id)
                else:
                    status = get_job_status_live(job_id)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(status).encode('utf-8'))
            except Exception as e:
                print(f"Error getting job status: {e}")
                self.send_error(500, str(e))
        else:
            self.send_error(404)
            
    def log_message(self, format, *args):
        # Suppress standard logging to keep daemon console clean
        pass

def main():
    print(f"Starting Quantum Presence Bridge (Target: {BACKEND_NAME})")
    print(f"Mode: {'MOCK' if MOCK_MODE else 'LIVE'}")
    
    # Start polling thread
    t = threading.Thread(target=telemetry_loop, daemon=True)
    t.start()
    
    # Start HTTP gateway
    try:
        server = HTTPServer(('127.0.0.1', PORT), QuantumGatewayHandler)
        print(f"Quantum HTTP Gateway listening on port {PORT}")
        server.serve_forever()
    except Exception as e:
        print(f"Failed to start HTTP server: {e}")

if __name__ == "__main__":
    main()
