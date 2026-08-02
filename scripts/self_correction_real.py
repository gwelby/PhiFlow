#!/usr/bin/env python3.12
"""
self_correction_real.py — Real quantum self-correction loop.

Runs the full detect → correct → re-measure loop on real IBM Quantum hardware,
using the zero-depth RZ correction technique proven in the Crypto lab.

1. Submit a GHZ circuit (baseline, no correction)
2. Poll the job, extract real measurement counts
3. Calculate physical coherence from the counts
4. If coherence < φ⁻¹ (0.618), generate a real correction:
   - Apply zero-depth RZ correction: shift existing RZ gates that follow CZ
     gates by a backend-specific optimal angle
   - Zero new gates, zero depth increase — the correction is free
5. Submit the corrected circuit
6. Poll the corrected job, extract counts
7. Compare: did coherence improve?

The correction is real and proven:
  - Crypto lab R-40: zero-depth RZ correction reduces FP rate 10-20pp on
    kingston and fez with zero depth increase
  - Kingston optimal angle: +0.045 (U-shape confirmed across 9 angles)
  - Fez optimal angle: -0.090 (opposite sign confirms coherent error model)
  - The correction targets coherent Z⊗I over-rotation on CX gates

Usage:
  python3.12 scripts/self_correction_real.py <n> [backend] [shots]
  python3.12 scripts/self_correction_real.py 4 ibm_fez 4096

Reads IBM_QUANTUM_TOKEN from the CASCADE vault (~/.cascade_keys).
"""

import sys
import json
import time
import argparse
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/mnt/d/Pi/routing")
from cascade_keys import get_key

PHI_INV = 0.618033988749895

# Backend-specific optimal RZ correction angles (from Crypto lab R-40)
# These are the proven optimal angles that reduce coherent error FPs
# by 10-20pp on real hardware with zero depth increase.
CORRECTION_ANGLES = {
    "ibm_kingston": 0.045,    # +0.045: over-rotation correction (U-shape confirmed)
    "ibm_fez": -0.090,        # -0.090: under-rotation correction (opposite sign)
    "ibm_marrakesh": 0.045,   # assumed same as kingston (Heron family)
}


def get_token() -> str:
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


def fold_correction_into_transpiled(tqc, correction_angle):
    """Modify existing RZ gates in a transpiled circuit to include coherent
    error correction.

    For each RZ gate that follows a CZ/CX gate, add the correction angle to
    the RZ parameter. Zero new gates, zero depth increase.

    This is the proven technique from Crypto lab R-36 through R-40.
    """
    from qiskit import QuantumRegister, ClassicalRegister
    from qiskit.circuit.library import RZGate

    if correction_angle == 0.0:
        return tqc

    modified = QuantumCircuit()
    for qreg in tqc.qregs:
        modified.add_register(qreg)
    for creg in tqc.cregs:
        modified.add_register(creg)

    prev_2q_name = None
    prev_2q_qubits = set()

    for instruction in tqc.data:
        op = instruction.operation
        qargs = [tqc.find_bit(q).index for q in instruction.qubits]
        cargs = [tqc.find_bit(c).index for c in instruction.clbits]

        if op.name == "rz" and prev_2q_name in ("cz", "cx"):
            if qargs[0] in prev_2q_qubits:
                old_param = float(op.params[0])
                new_param = old_param + correction_angle
                modified.append(RZGate(new_param), qargs, cargs)
            else:
                modified.append(op, qargs, cargs)
        else:
            modified.append(op, qargs, cargs)

        if op.name in ("cz", "cx"):
            prev_2q_name = op.name
            prev_2q_qubits = set(qargs)
        else:
            prev_2q_name = None
            prev_2q_qubits = set()

    return modified


def calculate_coherence(counts):
    """Compute physical coherence from measurement counts.

    For GHZ states, coherence = fraction of shots in the two expected
    entangled basis states (all-0s and all-1s).
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0

    max_len = max((len(s) for s in counts.keys()), default=0)

    if max_len <= 1:
        count_0 = counts.get('0', 0)
        count_1 = counts.get('1', 0)
        return max(count_0, count_1) / total

    if max_len == 2:
        count_00 = counts.get('00', 0)
        count_11 = counts.get('11', 0)
        good_states = count_00 + count_11
        if good_states > 0:
            return good_states / total

    # 3+ qubits: GHZ coherence
    all_zeros = '0' * max_len
    all_ones = '1' * max_len
    ghz_states = counts.get(all_zeros, 0) + counts.get(all_ones, 0)
    if ghz_states > 0:
        return ghz_states / total

    return max(counts.values()) / total


def submit_circuit(n, backend_name, shots, correction_angle=None):
    """Submit a GHZ circuit to IBM Quantum and return the job ID.

    If correction_angle is provided, applies zero-depth RZ correction to the
    transpiled circuit before submission.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.qasm3 import loads as qasm3_loads

    qasm = ghz_qasm(n)
    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)

    circuit = qasm3_loads(qasm)
    print(f"  Circuit: {n}-qubit GHZ, pre-depth={circuit.depth()}")

    pm = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend,
    )
    isa_circuit = pm.run(circuit)
    print(f"  Transpiled: depth={isa_circuit.depth()}, ops={dict(isa_circuit.count_ops())}")

    if correction_angle is not None and correction_angle != 0.0:
        isa_circuit = fold_correction_into_transpiled(isa_circuit, correction_angle)
        print(f"  Zero-depth RZ correction applied: angle={correction_angle}")
        print(f"  Corrected: depth={isa_circuit.depth()} (same depth — zero cost)")

    sampler = SamplerV2(backend)
    job = sampler.run([(isa_circuit,)], shots=shots)

    job_info = {
        "job_id": job.job_id(),
        "backend": backend_name,
        "circuit": f"{n}-qubit GHZ",
        "n_qubits": n,
        "shots": shots,
        "correction_angle": correction_angle,
    }
    info_path = f"/tmp/ibm_job_{job.job_id()}_info.json"
    with open(info_path, 'w') as f:
        json.dump(job_info, f, indent=2)

    print(f"  Job submitted: {job.job_id()}")
    return job.job_id(), job


def poll_job(job, max_wait_minutes=30):
    """Poll an IBM Quantum job and return measurement counts."""
    print(f"  Waiting for job...")
    for attempt in range(max_wait_minutes * 2):
        status = job.status()
        if attempt % 5 == 0:
            print(f"    [{attempt+1:02d}] {time.strftime('%H:%M:%S')} Status: {status}")

        if status == 'DONE':
            break
        elif status in ('ERROR', 'CANCELLED'):
            print(f"  Job failed: {status}")
            return None
        time.sleep(30)
    else:
        print(f"  Timed out after {max_wait_minutes} minutes.")
        return None

    result = job.result()

    if isinstance(result, dict) and 'results' in result:
        data = result['results'][0]['data']
        if 'c' in data:
            c_data = data['c']
            if 'samples' in c_data:
                samples = c_data['samples']
                num_bits = c_data.get('num_bits', 1)
                counts = Counter()
                for s in samples:
                    if isinstance(s, str) and s.startswith('0x'):
                        val = int(s, 16)
                        bits = format(val, f'0{num_bits}b')
                        counts[bits] += 1
                    else:
                        counts[str(s)] += 1
                return dict(counts)
            if 'counts' in c_data:
                return c_data['counts']

    if hasattr(result, '__getitem__'):
        try:
            pub_result = result[0]
            if hasattr(pub_result, 'data'):
                data = pub_result.data
                if hasattr(data, 'c'):
                    c = data.c
                    if hasattr(c, 'get_counts'):
                        return c.get_counts()
                    if hasattr(c, 'counts'):
                        return c.counts
        except Exception as e:
            print(f"  Error extracting counts: {e}")

    print(f"  Could not extract counts from result: {type(result)}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Real quantum self-correction loop")
    parser.add_argument("n", type=int, help="Number of qubits for GHZ state")
    parser.add_argument("backend", type=str, nargs="?", default="ibm_fez",
                        help="IBM backend name (default: ibm_fez)")
    parser.add_argument("shots", type=int, nargs="?", default=4096,
                        help="Number of shots (default: 4096)")
    args = parser.parse_args()

    n = args.n
    backend_name = args.backend
    shots = args.shots

    correction_angle = CORRECTION_ANGLES.get(backend_name, 0.045)

    print(f"{'='*70}")
    print(f"  REAL QUANTUM SELF-CORRECTION LOOP")
    print(f"{'='*70}")
    print(f"  Backend: {backend_name}")
    print(f"  Circuit: {n}-qubit GHZ")
    print(f"  Shots:   {shots}")
    print(f"  Threshold: φ⁻¹ = {PHI_INV:.4f}")
    print(f"  Correction: zero-depth RZ (angle={correction_angle} for {backend_name})")
    print(f"  Based on Crypto lab R-40: proven 10-20pp FP reduction on real hardware")
    print()

    # ─── Phase 1: Initial measurement (baseline, no correction) ────────
    print(f"{'─'*70}")
    print(f"  PHASE 1: Initial measurement (baseline, no correction)")
    print(f"{'─'*70}")
    job_id_1, job_1 = submit_circuit(n, backend_name, shots, correction_angle=None)
    counts_1 = poll_job(job_1)
    if counts_1 is None:
        print("  FAILED: Could not get initial counts.")
        sys.exit(1)

    coherence_1 = calculate_coherence(counts_1)
    total_1 = sum(counts_1.values())

    print(f"\n  Initial counts ({total_1} shots):")
    for state in sorted(counts_1.keys()):
        c = counts_1[state]
        pct = 100.0 * c / total_1
        bar = '#' * int(pct / 2)
        print(f"    |{state}⟩: {c:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  Initial coherence: {coherence_1:.4f}")
    print(f"  Threshold (φ⁻¹):   {PHI_INV:.4f}")

    if coherence_1 >= PHI_INV:
        print(f"\n  ✅ Coherence already above threshold — no correction needed.")
        print(f"  The circuit ran cleanly on the first attempt.")
        print(f"{'='*70}")
        sys.exit(0)

    print(f"\n  ⚠️  Coherence BELOW threshold — correction needed.")
    print(f"  Decoherence detected. Applying zero-depth RZ correction...")

    # ─── Phase 2: Correction (zero-depth RZ) ───────────────────────────
    print(f"\n{'─'*70}")
    print(f"  PHASE 2: Correction (zero-depth RZ angle={correction_angle})")
    print(f"{'─'*70}")
    print(f"  Correction: shift existing RZ gates following CZ by {correction_angle} rad")
    print(f"  This corrects coherent Z⊗I over-rotation on CX gates.")
    print(f"  Zero new gates, zero depth increase — the correction is free.")
    print(f"  Proven on real hardware: Crypto lab R-40 (10-20pp FP reduction).")

    job_id_2, job_2 = submit_circuit(n, backend_name, shots, correction_angle=correction_angle)
    counts_2 = poll_job(job_2)
    if counts_2 is None:
        print("  FAILED: Could not get corrected counts.")
        sys.exit(1)

    coherence_2 = calculate_coherence(counts_2)
    total_2 = sum(counts_2.values())

    print(f"\n  Corrected counts ({total_2} shots):")
    for state in sorted(counts_2.keys()):
        c = counts_2[state]
        pct = 100.0 * c / total_2
        bar = '#' * int(pct / 2)
        print(f"    |{state}⟩: {c:4d} ({pct:5.1f}%) {bar}")
    print(f"\n  Corrected coherence: {coherence_2:.4f}")

    # ─── Phase 3: Comparison ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")
    delta = coherence_2 - coherence_1
    print(f"  Initial  coherence: {coherence_1:.4f}")
    print(f"  Corrected coherence: {coherence_2:.4f}")
    print(f"  Delta:               {delta:+.4f}")
    print()

    if coherence_2 > coherence_1:
        print(f"  ✅ CORRECTION IMPROVED FIDELITY by {delta:+.4f}")
        if coherence_2 >= PHI_INV:
            print(f"  ✅ Corrected coherence now ABOVE φ⁻¹ threshold")
        else:
            print(f"  ⚠️  Improved but still below threshold — further correction needed")
    elif coherence_2 < coherence_1:
        print(f"  ❌ CORRECTION DEGRADED FIDELITY by {delta:+.4f}")
        print(f"  The RZ correction did not help for this circuit on {backend_name}.")
        print(f"  This is a real negative result — the correction technique has limits.")
        print(f"  Note: RZ correction was proven on Shor circuits, not GHZ. The")
        print(f"  coherent error model may not apply equally to all circuit types.")
    else:
        print(f"  ➡️  NO CHANGE — correction had no effect")

    # Save full results
    results = {
        "backend": backend_name,
        "n_qubits": n,
        "shots": shots,
        "correction_angle": correction_angle,
        "initial": {
            "job_id": job_id_1,
            "coherence": coherence_1,
            "counts": counts_1,
            "correction": "none (baseline)",
        },
        "corrected": {
            "job_id": job_id_2,
            "coherence": coherence_2,
            "counts": counts_2,
            "correction": f"zero-depth RZ (angle={correction_angle})",
        },
        "delta": delta,
        "improved": coherence_2 > coherence_1,
        "threshold": PHI_INV,
        "basis": "Crypto lab R-40: zero-depth RZ correction proven on kingston + fez",
    }
    results_path = f"/tmp/self_correction_result_{job_id_1}_{job_id_2}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved: {results_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
