#!/usr/bin/env python3.12
"""
Submit GHZ-6 crosstalk test jobs to IBM Quantum hardware.

Tests whether idle spectator qubits adjacent to the GHZ chain accelerate
coherence decay. Keeps the GHZ computation fixed (6 qubits, linear chain)
and varies the number of adjacent spectator qubits: 0, 2, 4, 5.

Uses a fixed heavy-hex layout:
  GHZ chain: physical qubits [4, 3, 16, 23, 24, 25]
  Available spectator qubits: [2, 5, 22, 26, 37]

Usage:
  python3.12 scripts/submit_ghz_crosstalk.py [backend] [shots]
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


# Fixed layout chosen from ibm_marrakesh heavy-hex coupling map.
# GHZ chain is a 6-qubit path; spectators are adjacent qubits not on the chain.
GHZ_PHYS = [4, 3, 16, 23, 24, 25]
SPECTATOR_PHYS = [2, 5, 22, 26, 37]


def ghz_crosstalk_qasm(n_ghz, n_spectators):
    """Return QASM for n_ghz GHZ state plus n_spectators idle qubits."""
    total = n_ghz + n_spectators
    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        "",
        f"qubit[{total}] q;",
        f"bit[{total}] c;",
        "",
        "    ry(0.5 * pi) q[0];",
    ]
    for i in range(n_ghz - 1):
        lines.append(f"    cx q[{i}], q[{i + 1}];")
    for i in range(total):
        lines.append(f"    c[{i}] = measure q[{i}];")
    return "\n".join(lines)


def submit_crosstalk_jobs(backend_name="ibm_marrakesh", shots=4096):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.qasm3 import loads as qasm3_loads

    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)

    # spectator counts to test: 0, 2, 4, 5 (max adjacent spectators available = 5)
    spectator_counts = [0, 2, 4, 5]
    job_ids = {}

    for k in spectator_counts:
        qasm = ghz_crosstalk_qasm(6, k)
        circuit = qasm3_loads(qasm)

        # Build initial layout: first 6 logical qubits -> GHZ_PHYS, next k -> first k SPECTATOR_PHYS
        initial_layout = list(GHZ_PHYS) + SPECTATOR_PHYS[:k]

        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=backend,
            initial_layout=initial_layout,
        )
        isa_circuit = pm.run(circuit)

        print(f"\n=== k={k} spectators ===")
        print(f"  Initial layout: {initial_layout}")
        print(f"  Transpiled depth: {isa_circuit.depth()}")
        print(f"  Transpiled ops: {dict(isa_circuit.count_ops())}")
        print(f"  Physical qubits used:")
        layout = isa_circuit.layout.initial_layout
        for i in range(6 + k):
            phys = layout[i]
            print(f"    logical q[{i}] -> physical {phys}")

        sampler = SamplerV2(backend)
        job = sampler.run([(isa_circuit,)], shots=shots)
        job_ids[k] = job.job_id()

        job_info = {
            "job_id": job.job_id(),
            "backend": backend_name,
            "circuit": f"GHZ-6 with {k} spectator qubits",
            "n_ghz": 6,
            "n_spectators": k,
            "shots": shots,
            "initial_layout": initial_layout,
            "transpiled_depth": isa_circuit.depth(),
            "transpiled_ops": dict(isa_circuit.count_ops()),
            "qasm": qasm,
        }
        info_path = f"/tmp/ibm_job_{job.job_id()}_info.json"
        with open(info_path, 'w') as f:
            json.dump(job_info, f, indent=2)
        print(f"  Job submitted: {job.job_id()}")
        print(f"  Info saved: {info_path}")

    print("\n" + "=" * 60)
    print("All crosstalk jobs submitted!")
    for k, job_id in sorted(job_ids.items()):
        print(f"  k={k}: {job_id}")
    print("=" * 60)
    return job_ids


if __name__ == '__main__':
    backend = sys.argv[1] if len(sys.argv) > 1 else "ibm_marrakesh"
    shots = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    submit_crosstalk_jobs(backend, shots)