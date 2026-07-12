#!/usr/bin/env python3.12
"""
Transpile a QASM program for a real IBM backend and emit a depth/layout report.

This is the PhiFlow quantum guardrail. It runs after QASM generation and before
any submission, warning the user if idle spectator qubits sit near active gates.

Usage:
  cat circuit.qasm | python3.12 scripts/transpile_report.py [backend_name]
  python3.12 scripts/transpile_report.py [backend_name] < circuit.qasm

Output: JSON report to stdout. Warnings go to stderr.
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


def report(qasm: str, backend_name: str = "ibm_marrakesh", optimization_level: int = 1):
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit import transpile
    from qiskit.qasm3 import loads as qasm3_loads
    from collections import defaultdict

    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)

    circuit = qasm3_loads(qasm)
    pre_depth = circuit.depth()
    pre_ops = dict(circuit.count_ops())

    isa_circuit = transpile(
        circuit,
        backend,
        optimization_level=optimization_level,
        seed_transpiler=42,
    )

    post_depth = isa_circuit.depth()
    post_ops = dict(isa_circuit.count_ops())

    # Physical qubits used by the logical circuit qubits, in order.
    layout = [
        isa_circuit.find_bit(isa_circuit.qubits[i]).index
        for i in range(circuit.num_qubits)
    ]

    # Build adjacency list from coupling map and find idle neighbors (spectators).
    coupling = backend.configuration().coupling_map
    adj = defaultdict(set)
    for a, b in coupling:
        adj[a].add(b)
        adj[b].add(a)

    layout_set = set(layout)
    spectator_qubits = set()
    for phys in layout:
        for nb in adj[phys]:
            if nb not in layout_set:
                spectator_qubits.add(nb)

    spectator_qubits = sorted(spectator_qubits)
    spectator_count = len(spectator_qubits)

    result = {
        "backend": backend_name,
        "num_logical_qubits": circuit.num_qubits,
        "pre_depth": pre_depth,
        "post_depth": post_depth,
        "pre_ops": pre_ops,
        "post_ops": post_ops,
        "layout": layout,
        "spectator_qubits": spectator_qubits,
        "spectator_count": spectator_count,
    }

    # Warning logic
    if spectator_count > 0:
        result["warning"] = (
            f"{spectator_count} adjacent idle spectator qubit(s) detected near "
            f"active gates on {backend_name}; crosstalk risk. "
            f"GHZ crosstalk test showed ~50% spectator error and ~47% coherence drop "
            f"with 2 spectators."
        )
    elif post_depth > 100:
        result["warning"] = (
            f"Transpiled depth ({post_depth}) is high; coherence may decay on "
            f"NISQ hardware. Consider a shallower decomposition or better layout."
        )
    else:
        result["warning"] = None

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python3.12 transpile_report.py <qasm_file> [backend]", file=sys.stderr)
        sys.exit(1)

    qasm_path = Path(sys.argv[1])
    backend = sys.argv[2] if len(sys.argv) > 2 else "ibm_marrakesh"

    if not qasm_path.exists():
        print(f"QASM file not found: {qasm_path}", file=sys.stderr)
        sys.exit(1)

    qasm = qasm_path.read_text()
    if not qasm.strip():
        print("QASM file is empty.", file=sys.stderr)
        sys.exit(1)

    result = report(qasm, backend)
    print(json.dumps(result, indent=2))

    if result.get("warning"):
        print(f"\n⚠️  {result['warning']}", file=sys.stderr)


if __name__ == '__main__':
    main()