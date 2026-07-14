#!/usr/bin/env python3.12
"""
Fetch a backend topology profile from IBM Quantum using IBM_QUANTUM_TOKEN.

This is the Python bridge for `phic --topology-aware`. It uses the same
credential (IBM_QUANTUM_TOKEN from ~/.cascade_keys) as the rest of the
PhiFlow IBM pipeline, avoiding the need for a separate IBM Cloud IAM
API key + service CRN.

Emits a JSON object matching the Rust `BackendTopologyProfile` struct:
{
  "backend_name": "ibm_marrakesh",
  "family": "heron",
  "num_qubits": 156,
  "coupling_map": [[0,1], [1,2], ...],
  "native_two_qubit_gate": "cz",
  "qubits": {
    "0": {"t1_s": 0.0003, "t2_s": 0.0002, "readout_error": 0.01},
    ...
  },
  "edges": {
    "0,1": {"duration_s": 0.0000005, "error": 0.001},
    ...
  }
}

Usage:
  python3.12 scripts/fetch_topology_profile.py <backend_name>
  python3.12 scripts/fetch_topology_profile.py ibm_marrakesh
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, "/mnt/d/Pi/routing")
from cascade_keys import get_key


def get_token() -> str:
    token = get_key("IBM_QUANTUM_TOKEN")
    if not token:
        raise KeyError("IBM_QUANTUM_TOKEN not found in ~/.cascade_keys")
    return token


def infer_family(backend_name: str, backend) -> str:
    """Infer processor family from backend name or config."""
    name_lower = backend_name.lower()
    # Check backend configuration for processor type
    try:
        config = backend.configuration()
        if hasattr(config, "processor_type"):
            pt = config.processor_type
            if isinstance(pt, dict):
                family = pt.get("family", "").lower()
                if "heron" in family or "fez" in family:
                    return "heron"
                if "eagle" in family:
                    return "eagle"
    except Exception:
        pass
    # Fallback: name-based heuristic
    if "heron" in name_lower or "fez" in name_lower or "marrakesh" in name_lower:
        return "heron"
    if "eagle" in name_lower:
        return "eagle"
    return "unknown"


def infer_native_gate(backend) -> str:
    """Infer native two-qubit gate from basis gates."""
    try:
        config = backend.configuration()
        basis_gates = [g.lower() for g in config.basis_gates]
        if "cz" in basis_gates:
            return "cz"
        if "ecr" in basis_gates:
            return "ecr"
    except Exception:
        pass
    # Default for Heron
    return "cz"


def extract_coupling_map(backend) -> list:
    """Extract coupling map as list of [a, b] pairs."""
    try:
        cmap = backend.configuration().coupling_map
        return [[a, b] for a, b in cmap]
    except Exception:
        return []


def extract_num_qubits(backend) -> int:
    """Extract number of qubits."""
    try:
        return backend.configuration().n_qubits
    except Exception:
        return 0


def extract_qubit_calibrations(backend) -> dict:
    """Extract per-qubit calibration data."""
    qubits = {}
    try:
        props = backend.properties()
        if props is None:
            return qubits
        n_qubits = backend.configuration().n_qubits
        for q in range(n_qubits):
            cal = {}
            try:
                t1 = props.t1(q)
                if t1 is not None:
                    cal["t1_s"] = t1
            except Exception:
                pass
            try:
                t2 = props.t2(q)
                if t2 is not None:
                    cal["t2_s"] = t2
            except Exception:
                pass
            try:
                readout = props.readout_error(q)
                if readout is not None:
                    cal["readout_error"] = readout
            except Exception:
                pass
            if cal:
                qubits[str(q)] = cal
    except Exception:
        pass
    return qubits


def extract_edge_calibrations(backend) -> list:
    """Extract per-edge calibration data as a list of {edge: [a,b], ...} dicts."""
    edges = []
    try:
        props = backend.properties()
        if props is None:
            return edges
        cmap = backend.configuration().coupling_map
        for a, b in cmap:
            entry = {"edge": [min(a, b), max(a, b)]}
            try:
                gate_error = props.gate_error("cz", [a, b])
                if gate_error is not None:
                    entry["error"] = gate_error
            except Exception:
                pass
            try:
                gate_length = props.gate_length("cz", [a, b])
                if gate_length is not None:
                    entry["duration_s"] = gate_length
            except Exception:
                pass
            if len(entry) > 1:  # has more than just "edge"
                edges.append(entry)
    except Exception:
        pass
    return edges


def fetch_topology_profile(backend_name: str) -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)

    profile = {
        "backend_name": backend_name,
        "family": infer_family(backend_name, backend),
        "num_qubits": extract_num_qubits(backend),
        "coupling_map": extract_coupling_map(backend),
        "native_two_qubit_gate": infer_native_gate(backend),
        "qubits": extract_qubit_calibrations(backend),
        "edges": extract_edge_calibrations(backend),
    }
    return profile


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3.12 scripts/fetch_topology_profile.py <backend_name>", file=sys.stderr)
        sys.exit(1)
    backend_name = sys.argv[1]
    try:
        profile = fetch_topology_profile(backend_name)
        print(json.dumps(profile))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
