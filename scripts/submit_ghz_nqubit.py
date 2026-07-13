#!/usr/bin/env python3.12
"""
Submit an N-qubit GHZ circuit to IBM Quantum hardware.
Generates QASM for any n >= 2, transpiles, and submits.

Usage:
  python3.12 submit_ghz_nqubit.py <n> [backend] [shots] [--layout-aware]
  python3.12 submit_ghz_nqubit.py 6 ibm_marrakesh 4096 --layout-aware
"""
import sys
import json
import argparse
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


def build_coupling_map(backend):
    """Return adjacency list for the backend coupling graph."""
    cmap = backend.configuration().coupling_map
    adj = {}
    for src, dst in cmap:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    return adj


def count_spectators(path, adj):
    """Count adjacent idle spectator qubits for a physical qubit path."""
    path_set = set(path)
    total = 0
    for q in path:
        for neighbor in adj.get(q, set()):
            if neighbor not in path_set:
                total += 1
    return total


def find_best_ghz_path(n, adj, max_candidates=10000):
    """
    Find a simple path of n connected qubits with the fewest adjacent idle
    spectators. GHZ circuits are linear chains, so we want a path on the
    device coupling graph. Returns the best path and its spectator count.
    """
    best_path = None
    best_spectators = float('inf')
    candidates = 0

    def dfs(path, visited):
        nonlocal best_path, best_spectators, candidates
        if candidates >= max_candidates:
            return
        if len(path) == n:
            candidates += 1
            spectators = count_spectators(path, adj)
            if spectators < best_spectators:
                best_spectators = spectators
                best_path = list(path)
            return
        current = path[-1]
        for neighbor in sorted(adj.get(current, set())):
            if neighbor in visited:
                continue
            if best_spectators != float('inf'):
                # Pruning: even if every remaining qubit adds 0 spectators,
                # we can't beat best. But more importantly, if we already have
                # more spectators than best possible, prune. This is a simple
                # depth-first without a tight lower bound, so we just continue.
                pass
            visited.add(neighbor)
            path.append(neighbor)
            dfs(path, visited)
            path.pop()
            visited.remove(neighbor)

    for start in sorted(adj.keys()):
        dfs([start], {start})

    return best_path, best_spectators


def submit_ghz(n, backend_name="ibm_marrakesh", shots=4096, layout_aware=False):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.qasm3 import loads as qasm3_loads
    from qiskit.transpiler import Layout

    qasm = ghz_qasm(n)
    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)
    num_device_qubits = backend.configuration().n_qubits

    circuit = qasm3_loads(qasm)
    print(f"Circuit: {n}-qubit GHZ, pre-depth={circuit.depth()}, ops={dict(circuit.count_ops())}")

    initial_layout = None
    layout_metadata = {"mode": "default"}

    if layout_aware:
        adj = build_coupling_map(backend)
        path, spectators = find_best_ghz_path(n, adj)
        if path is None:
            raise RuntimeError(f"No connected path of {n} qubits found on {backend_name}")
        print(f"Layout-aware path: {path} (spectators={spectators})")
        # Map virtual qubit i -> physical qubit path[i]
        layout_mapping = {circuit.qubits[i]: path[i] for i in range(n)}
        initial_layout = Layout(layout_mapping)
        layout_metadata = {
            "mode": "layout-aware",
            "path": path,
            "spectators": spectators,
        }

    pm = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
        initial_layout=initial_layout,
    )
    isa_circuit = pm.run(circuit)
    print(f"Transpiled: depth={isa_circuit.depth()}, ops={dict(isa_circuit.count_ops())}")

    if args.dry_run:
        print("Dry run: no job submitted.")
        return None

    sampler = SamplerV2(backend)
    job = sampler.run([(isa_circuit,)], shots=shots)

    job_info = {
        "job_id": job.job_id(),
        "backend": backend_name,
        "circuit": f"{n}-qubit GHZ",
        "n_qubits": n,
        "shots": shots,
        "qasm": qasm,
        "layout": layout_metadata,
        "device_qubits": num_device_qubits,
    }
    info_path = f"/tmp/ibm_job_{job.job_id()}_info.json"
    with open(info_path, 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"\nJob submitted: {job.job_id()}")
    print(f"  Info saved: {info_path}")
    print(f"  Poll with: python3.12 scripts/poll_ibm_real.py {job.job_id()}")
    return job.job_id()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit n-qubit GHZ circuit to IBM Quantum")
    parser.add_argument("n", type=int, help="Number of qubits")
    parser.add_argument("backend", nargs="?", default="ibm_marrakesh", help="Backend name")
    parser.add_argument("shots", nargs="?", type=int, default=4096, help="Shots")
    parser.add_argument("--layout-aware", action="store_true", help="Use layout-aware qubit selection")
    parser.add_argument("--dry-run", action="store_true", help="Find layout and transpile but do not submit")
    args = parser.parse_args()
    submit_ghz(args.n, args.backend, args.shots, args.layout_aware)
