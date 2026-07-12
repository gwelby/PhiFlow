#!/usr/bin/env python3.12
"""
Submit an N-qubit GHZ circuit to IBM Quantum hardware.
Generates QASM for any n >= 2, transpiles, and submits.
Usage:
  python3.12 submit_ghz_nqubit.py <n> [backend] [shots]
  python3.12 submit_ghz_nqubit.py 6 ibm_marrakesh 4096
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, "/mnt/d/Pi/routing")
from cascade_keys import get_key



def get_token() -> str:
    """Read IBM_QUANTUM_TOKEN from the CASCADE vault via cascade_keys."""
    token = get_key("IBM_QUANTUM_TOKEN")
    if not token:
        raise KeyError("IBM_QUANTUM_TOKEN not found in vault")
    return token


def ghz_qasm(n):
    """Return an OpenQASM 3.0 program for an n-qubit GHZ state."""
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        f"qubit[{n}] q;",
        f"bit[{n}] c;",
        "",
        "    ry(0.5 * pi) q[0];",
    ]
    for i in range(n - 1):
        lines.append(f"    cx q[{i}], q[{i + 1}];")
    for i in range(n):
        lines.append(f"    c[{i}] = measure q[{i}];")
    return "\n".join(lines)


def submit_ghz(n, backend_name="ibm_marrakesh", shots=4096):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.qasm3 import loads as qasm3_loads

    qasm = ghz_qasm(n)
    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)

    circuit = qasm3_loads(qasm)
    print(f"Circuit: {n}-qubit GHZ, pre-depth={circuit.depth()}, ops={dict(circuit.count_ops())}")

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(circuit)
    print(f"Transpiled: depth={isa_circuit.depth()}, ops={dict(isa_circuit.count_ops())}")

    sampler = SamplerV2(backend)
    job = sampler.run([(isa_circuit,)], shots=shots)

    job_info = {
        "job_id": job.job_id(),
        "backend": backend_name,
        "circuit": f"{n}-qubit GHZ",
        "n_qubits": n,
        "shots": shots,
        "qasm": qasm,
    }
    info_path = f"/tmp/ibm_job_{job.job_id()}_info.json"
    with open(info_path, 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"\nJob submitted: {job.job_id()}")
    print(f"  Info saved: {info_path}")
    print(f"  Poll with: python3.12 scripts/poll_ibm_real.py {job.job_id()}")
    return job.job_id()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3.12 submit_ghz_nqubit.py <n> [backend] [shots]")
        sys.exit(1)
    n = int(sys.argv[1])
    backend = sys.argv[2] if len(sys.argv) > 2 else "ibm_marrakesh"
    shots = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    submit_ghz(n, backend, shots)