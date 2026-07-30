#!/usr/bin/env python3
"""
H_prod Quantum SWAP Test — God Equation Statistical Independence Verification
==============================================================================
Hermes | May 24, 2026 | Target: ibm_fez (156-qubit IBM Quantum)

THE PHYSICS QUESTION:
The God Equation (λ_c from l_P, 1.48% error, CONDITIONAL 0.88) depends on H_prod:
statistical independence of the three steps of the ℤ₃ closure walk.

Classically, we can only show zero covariance — which is WEAKER than independence.
Quantum mechanically, we can encode the walk states as quantum amplitudes and use
a SWAP test to directly measure the fidelity between:

  |ψ_actual⟩  = amplitude encoding of actual 3-step distribution T³|start⟩
  |ψ_product⟩ = amplitude encoding of the product-of-independent-steps distribution

If fidelity ≈ 1.0 → steps are statistically independent → H_prod holds.
If fidelity < 1.0 → steps are correlated → we measure exactly how much.

THE ℤ₃ WALK:
The general ℤ₃-circulant transition matrix is T = a·S̄ + b·S̄² where:
  S̄|0⟩=|1⟩, S̄|1⟩=|2⟩, S̄|2⟩=|0⟩ (shift operator on ℤ₃)
  a + b = 1 (stochasticity)

Key cases:
  T_sym  (a=b=1/2): nearest-neighbor symmetric walk — T³ not diagonal
  T_shift (a=1,b=0): pure shift — S̄³=I, deterministic return
  T_chiral:       the projected operator from the chiral ℤ₃ Lagrangian

The SWAP test measures fidelity between actual and factorized distributions
for ANY (a,b) — including the physical operators the God Equation depends on.

CIRCUIT DESIGN (5 qubits):
  q0,q1 : |ψ_actual⟩  — 2-qubit encoding of 3-level actual distribution
  q2,q3 : |ψ_product⟩ — 2-qubit encoding of 3-level product distribution  
  q4    : SWAP test ancilla — measured in X-basis; P(|0⟩) = (1+F)/2

USAGE:
  python3 hprod_swap_test.py --backend ibm_fez --a 0.5 --b 0.5
  python3 hprod_swap_test.py --backend ibm_fez --a 1.0 --b 0.0  # pure shift
  python3 hprod_swap_test.py --scan  # scan (a,b) parameter space locally
"""

import argparse
import json
import os
import sys
import numpy as np
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# PART 1: CLASSICAL ℤ₃ WALK DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════

def z3_transition_matrix(a: float, b: float = None) -> np.ndarray:
    """Build ℤ₃ circulant transition matrix T = a·S̄ + b·S̄².
    If b is None, b = 1 - a (stochastic normalization)."""
    if b is None:
        b = 1.0 - a
    # S̄ (shift) and S̄² (reverse shift) on ℤ₃
    S  = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]], dtype=float)
    S2 = np.array([[0, 0, 1],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=float)
    return a * S + b * S2


def actual_3step_distribution(T: np.ndarray, start_state: int = 0) -> np.ndarray:
    """Distribution after three ACTUAL (correlated) steps: T³|start⟩."""
    T3 = np.linalg.matrix_power(T, 3)
    start_vec = np.zeros(3)
    start_vec[start_state] = 1.0
    return T3 @ start_vec


def product_3step_distribution(T: np.ndarray, start_state: int = 0) -> np.ndarray:
    """Distribution if three steps were INDEPENDENT.
    
    If each step is an independent draw from the single-step distribution 
    p₁(k) = ⟨k|T|start⟩, then after 3 independent steps, the distribution
    over final states is the 3-fold convolution of p₁ with itself.
    
    For ℤ₃ (cyclic group of order 3), this is computed via the group convolution.
    """
    p1 = T[start_state]  # single-step distribution from start_state
    
    # 3-fold convolution on ℤ₃: P(k) = Σ_{i+j+l≡k mod 3} p₁(i)·p₁(j)·p₁(l)
    p3 = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for l_ in range(3):
                k = (i + j + l_) % 3
                p3[k] += p1[i] * p1[j] * p1[l_]
    return p3


def distribution_distance(p: np.ndarray, q: np.ndarray) -> dict:
    """Compute multiple distance metrics between two distributions."""
    # Total variation distance
    tv = 0.5 * np.sum(np.abs(p - q))
    # Hellinger distance
    hellinger = np.sqrt(1.0 - np.sum(np.sqrt(p * q)))
    # Classical fidelity (Bhattacharyya coefficient)
    fidelity = np.sum(np.sqrt(p * q))
    # Jensen-Shannon divergence
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * np.log(a / b, where=(a > 0) & (b > 0)))
    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    
    return {
        "total_variation": float(tv),
        "hellinger": float(hellinger),
        "classical_fidelity": float(fidelity),
        "jensen_shannon": float(js),
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 2: QUANTUM STATE PREPARATION
# ═══════════════════════════════════════════════════════════════════════

def distribution_to_amplitudes(p: np.ndarray) -> np.ndarray:
    """Convert a 3-element probability distribution to 2-qubit quantum amplitudes.
    
    Maps: p[0]→|00⟩, p[1]→|01⟩, p[2]→|10⟩, with |11⟩ unused (zero amplitude).
    The state is: √p[0]|00⟩ + √p[1]|01⟩ + √p[2]|10⟩
    """
    amps = np.zeros(4, dtype=complex)
    amps[0] = np.sqrt(p[0])
    amps[1] = np.sqrt(p[1])
    amps[2] = np.sqrt(p[2])
    amps[3] = 0.0  # |11⟩ unused
    # Normalize (handles floating point)
    norm = np.sqrt(np.sum(np.abs(amps)**2))
    if norm > 0:
        amps /= norm
    return amps


def state_preparation_circuit(qc, qubits, amplitudes):
    """Add state preparation for 2-qubit amplitude state to quantum circuit.
    
    Uses the method from Shende, Bullock, Markov (2004) for arbitrary
    2-qubit state preparation via Schmidt decomposition.
    """
    from qiskit import QuantumCircuit
    import qiskit.quantum_info as qi
    
    # Create the statevector
    state = qi.Statevector(amplitudes)
    # Use Qiskit's initialize (handles arbitrary state prep)
    qc.initialize(amplitudes, qubits)
    return qc


def build_swap_test_circuit(p_actual: np.ndarray, p_product: np.ndarray) -> 'QuantumCircuit':
    """Build the full SWAP test circuit comparing actual vs product distributions.
    
    Architecture (5 qubits, 1 classical bit):
      q0,q1 : |ψ_actual⟩  — actual 3-step distribution as quantum state
      q2,q3 : |ψ_product⟩ — product (independent) distribution as quantum state
      q4    : SWAP ancilla — Hadamard → controlled-SWAP → Hadamard → measure
      c0    : classical register for ancilla measurement
    """
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    
    qr = QuantumRegister(5, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Step 1: Prepare the two quantum states
    amps_actual = distribution_to_amplitudes(p_actual)
    amps_product = distribution_to_amplitudes(p_product)
    
    qc.initialize(amps_actual, [qr[0], qr[1]])
    qc.initialize(amps_product, [qr[2], qr[3]])
    qc.barrier()
    
    # Step 2: SWAP test on the ancilla
    qc.h(qr[4])  # Ancilla in superposition
    
    # Controlled-SWAP between the two 2-qubit registers
    # CSWAP(q4 controls swap of q0↔q2)
    qc.cswap(qr[4], qr[0], qr[2])
    qc.cswap(qr[4], qr[1], qr[3])
    
    qc.h(qr[4])  # Second Hadamard
    qc.barrier()
    
    # Step 3: Measure ancilla
    qc.measure(qr[4], cr[0])
    
    return qc


# ═══════════════════════════════════════════════════════════════════════
# PART 3: EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def analyze_classical(a: float, b: float = None, start_state: int = 0):
    """Classical analysis: compute actual vs product distributions and distances."""
    T = z3_transition_matrix(a, b)
    p_actual = actual_3step_distribution(T, start_state)
    p_product = product_3step_distribution(T, start_state)
    distances = distribution_distance(p_actual, p_product)
    
    # Eigenvalues of T for diagnostics
    eigenvalues = np.linalg.eigvals(T)
    
    result = {
        "parameters": {"a": float(a), "b": float(1-a) if b is None else float(b), "start_state": start_state},
        "T_matrix": T.tolist(),
        "T3_matrix": np.linalg.matrix_power(T, 3).tolist(),
        "T_eigenvalues": [complex(x.real, x.imag) for x in eigenvalues],
        "actual_distribution": p_actual.tolist(),
        "product_distribution": p_product.tolist(),
        "distances": distances,
        "h_prod_verdict": "INDEPENDENT" if distances["total_variation"] < 0.01 else "CORRELATED",
        "h_prod_confidence": float(1.0 - distances["total_variation"]),
    }
    return result


def run_local_simulation(a: float, b: float = None, shots: int = 8192):
    """Run SWAP test on local simulator."""
    from qiskit_aer import AerSimulator
    
    if b is None:
        b = 1.0 - a
    
    T = z3_transition_matrix(a, b)
    p_actual = actual_3step_distribution(T)
    p_product = product_3step_distribution(T)
    
    qc = build_swap_test_circuit(p_actual, p_product)
    
    simulator = AerSimulator()
    from qiskit import transpile
    qc_compiled = transpile(qc, simulator)
    job = simulator.run(qc_compiled, shots=shots)
    counts = job.result().get_counts()
    
    # P(|0⟩) from ancilla measurement → fidelity F = 2*P(0) - 1
    p0 = counts.get('0', 0) / shots
    quantum_fidelity = max(2 * p0 - 1, 0.0)  # clamp negative to 0
    
    classical_analysis = analyze_classical(a, b)
    
    return {
        **classical_analysis,
        "quantum_swap_test": {
            "shots": shots,
            "counts": {k: int(v) for k, v in counts.items()},
            "p_zero": float(p0),
            "quantum_fidelity": float(quantum_fidelity),
        },
        "comparison": {
            "quantum_fidelity": float(quantum_fidelity),
            "classical_fidelity": classical_analysis["distances"]["classical_fidelity"],
            "agreement": "CONSISTENT" if abs(quantum_fidelity - classical_analysis["distances"]["classical_fidelity"]) < 0.1 else "DIVERGENT",
        }
    }


def run_ibm_hardware(a: float, b: float = None, backend_name: str = "ibm_fez", 
                     token: str = None, shots: int = 4096):
    """Run SWAP test on real IBM Quantum hardware."""
    from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2
    from qiskit import transpile
    
    token = token or os.environ.get('IBM_QUANTUM_TOKEN', '')
    if not token:
        raise ValueError("IBM_QUANTUM_TOKEN required. Set env var or pass --token.")
    
    if b is None:
        b = 1.0 - a
    
    T = z3_transition_matrix(a, b)
    p_actual = actual_3step_distribution(T)
    p_product = product_3step_distribution(T)
    
    qc = build_swap_test_circuit(p_actual, p_product)
    
    print(f"Connecting to IBM Quantum...")
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    
    print(f"Selecting backend: {backend_name}")
    backend = service.backend(backend_name)
    
    print(f"Transpiling for {backend_name}...")
    qc_transpiled = transpile(qc, backend=backend, optimization_level=3)
    
    print(f"Submitting job to {backend_name} ({backend.num_qubits} qubits)...")
    
    with Session(service=service, backend=backend) as session:
        sampler = SamplerV2(session=session)
        job = sampler.run([qc_transpiled], shots=shots)
        print(f"Job ID: {job.job_id()}")
        print(f"Waiting for results...")
        result = job.result()
    
    # Extract results
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()
    
    p0 = counts.get(0, 0) / shots
    quantum_fidelity = max(2 * p0 - 1, 0.0)
    
    classical_analysis = analyze_classical(a, b)
    
    return {
        **classical_analysis,
        "quantum_swap_test": {
            "backend": backend_name,
            "job_id": job.job_id(),
            "shots": shots,
            "counts": {int(k): int(v) for k, v in counts.items()},
            "p_zero": float(p0),
            "quantum_fidelity": float(quantum_fidelity),
        },
        "comparison": {
            "quantum_fidelity": float(quantum_fidelity),
            "classical_fidelity": classical_analysis["distances"]["classical_fidelity"],
        },
        "timestamp": datetime.now().isoformat(),
    }


def scan_parameter_space(backend: str = None, token: str = None):
    """Scan (a,b) parameter space to find where H_prod holds vs fails."""
    results = []
    
    # Key points in parameter space
    points = [
        (1.0, 0.0, "pure_shift"),
        (0.0, 1.0, "pure_reverse_shift"),
        (0.5, 0.5, "symmetric_nearest_neighbor"),
        (0.75, 0.25, "chiral_asymmetric"),
        (0.25, 0.75, "anti_chiral"),
        (0.9, 0.1, "near_pure_shift"),
        (0.1, 0.9, "near_pure_reverse"),
    ]
    
    for a, b, label in points:
        result = analyze_classical(a, b)
        result["label"] = label
        results.append(result)
        verdict = result["h_prod_verdict"]
        fidelity = result["distances"]["classical_fidelity"]
        tv = result["distances"]["total_variation"]
        print(f"  {label:25s}  a={a:.2f} b={b:.2f}  TV={tv:.6f}  Fidelity={fidelity:.6f}  → {verdict}")
    
    # Find the boundary where independence breaks
    print("\n--- Parameter Scan (fine grid) ---")
    scan_results = []
    for a in np.linspace(0, 1, 21):
        b = 1.0 - a
        r = analyze_classical(a, b)
        scan_results.append({
            "a": float(a), "b": float(b),
            "tv": r["distances"]["total_variation"],
            "fidelity": r["distances"]["classical_fidelity"],
            "verdict": r["h_prod_verdict"],
        })
    
    # Sort by fidelity
    scan_results.sort(key=lambda x: x["fidelity"], reverse=True)
    
    print("\nTop 5 (closest to independence):")
    for r in scan_results[:5]:
        print(f"  a={r['a']:.3f} b={r['b']:.3f}  TV={r['tv']:.6f}  Fidelity={r['fidelity']:.6f}  → {r['verdict']}")
    
    return {
        "key_points": results,
        "fine_scan": scan_results,
        "optimal": scan_results[0],
        "conclusion": "H_prod holds only if fidelity > 0.99 at the physical (a,b) values."
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="H_prod Quantum SWAP Test — God Equation Statistical Independence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 hprod_swap_test.py --scan                    # Full parameter scan (local)
  python3 hprod_swap_test.py --a 0.5 --b 0.5           # Symmetric walk, local sim
  python3 hprod_swap_test.py --a 1.0 --b 0.0           # Pure shift, local sim
  python3 hprod_swap_test.py --a 0.5 --backend ibm_fez # Symmetric walk on REAL HARDWARE
  python3 hprod_swap_test.py --classical-only --scan   # Classical analysis only
        """
    )
    parser.add_argument('--a', type=float, default=0.5, help='Circulant parameter a (default: 0.5)')
    parser.add_argument('--b', type=float, default=None, help='Circulant parameter b (default: 1-a)')
    parser.add_argument('--backend', type=str, default=None, 
                        help='IBM Quantum backend (e.g., ibm_fez). Omit for local simulator.')
    parser.add_argument('--token', type=str, default=None,
                        help='IBM Quantum API token (or set IBM_QUANTUM_TOKEN env var)')
    parser.add_argument('--shots', type=int, default=8192, 
                        help='Measurement shots (default: 8192 for sim, 4096 for hardware)')
    parser.add_argument('--scan', action='store_true', 
                        help='Full parameter space scan')
    parser.add_argument('--classical-only', action='store_true',
                        help='Classical analysis only (no quantum circuit)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("H_prod QUANTUM SWAP TEST — God Equation Independence Verification")
    print("=" * 70)
    print(f"Target: {args.backend or 'local simulator'}")
    print(f"Parameters: a={args.a}, b={args.b or (1-args.a):.3f}")
    print()
    
    if args.scan:
        result = scan_parameter_space(args.backend, args.token)
    elif args.classical_only:
        result = analyze_classical(args.a, args.b)
        print(f"Actual distribution:   {result['actual_distribution']}")
        print(f"Product distribution:  {result['product_distribution']}")
        print(f"Total variation:       {result['distances']['total_variation']:.6f}")
        print(f"Classical fidelity:    {result['distances']['classical_fidelity']:.6f}")
        print(f"Jensen-Shannon:        {result['distances']['jensen_shannon']:.6f}")
        print(f"\nVERDICT: {result['h_prod_verdict']} (confidence: {result['h_prod_confidence']:.4f})")
    elif args.backend:
        shots = min(args.shots, 4096)  # Hardware limit
        print(f"⚠️  RUNNING ON REAL IBM QUANTUM HARDWARE: {args.backend}")
        print(f"   This will consume quantum compute time.")
        result = run_ibm_hardware(args.a, args.b, args.backend, args.token, shots)
        print(f"\nJob ID: {result['quantum_swap_test']['job_id']}")
        print(f"Quantum fidelity:  {result['comparison']['quantum_fidelity']:.6f}")
        print(f"Classical fidelity: {result['comparison']['classical_fidelity']:.6f}")
    else:
        result = run_local_simulation(args.a, args.b, args.shots)
        print(f"Quantum fidelity:  {result['comparison']['quantum_fidelity']:.6f}")
        print(f"Classical fidelity: {result['comparison']['classical_fidelity']:.6f}")
        print(f"Agreement:         {result['comparison']['agreement']}")
        print(f"VERDICT:           {result['h_prod_verdict']}")
    
    # Save output
    output_path = args.output or f"/mnt/d/Projects/PUBLISHING/outputs/hprod_swap_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
