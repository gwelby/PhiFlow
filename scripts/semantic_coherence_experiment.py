#!/usr/bin/env python3.12
"""
Semantic Coherence as Quantum Fidelity Predictor — Experiment Script

Compiles four PhiFlow programs with different semantic structures to QASM,
runs each on the Qiskit Aer simulator (ideal) and IBM Quantum hardware (real),
and compares the measurement distributions.

Usage:
  python3.12 scripts/semantic_coherence_experiment.py --sim-only
  python3.12 scripts/semantic_coherence_experiment.py --backend ibm_marrakesh --shots 4096
  python3.12 scripts/semantic_coherence_experiment.py --backend ibm_marrakesh --poll <job_ids...>

Programs:
  1. coherent_council   — 6 intentions, all resonate 0.85, all on 432 Hz (unified)
  2. dissonant_council  — 6 intentions, alternating 0.85/0.15, all on 432 Hz (divided)
  3. polarized_council  — 6 intentions, TEAM_A/TEAM_B alternating, all on 432 Hz (polarized)
  4. chambered_council  — 6 intentions, 3 on 432 Hz + 3 on 528 Hz (two chambers)

Hypothesis: PhiFlow's semantic constructs (resonance values, team directions,
frequency channels) produce quantum circuits with measurably different physical
behavior on real hardware. The frequency channel construct in particular
generates different circuit topology (not just different rotation angles),
which should produce significantly different hardware fidelity.
"""
import sys
import json
import argparse
import subprocess
import math
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/mnt/d/Pi/routing")
from cascade_keys import get_key

PHIFLOW_ROOT = Path("/mnt/d/Projects/phiflow")
EXPERIMENT_DIR = PHIFLOW_ROOT / "examples" / "experiment"
REPORTS_DIR = PHIFLOW_ROOT / "reports"
PHIC = PHIFLOW_ROOT / "target" / "release" / "phic"

PROGRAMS = [
    {
        "name": "coherent_council",
        "file": "coherent_council.phi",
        "semantic": "unified — all agree on 432 Hz",
        "expected_circuit": "6-qubit GHZ chain, uniform ry(0.85π), 5 CX gates",
    },
    {
        "name": "dissonant_council",
        "file": "dissonant_council.phi",
        "semantic": "divided — alternating 0.85/0.15 on 432 Hz",
        "expected_circuit": "6-qubit entangled, alternating ry(0.85π)/ry(0.15π), 5 CX gates",
    },
    {
        "name": "polarized_council",
        "file": "polarized_council.phi",
        "semantic": "polarized — TEAM_A vs TEAM_B on 432 Hz",
        "expected_circuit": "6-qubit entangled, complementary rotations, 5 CX gates",
    },
    {
        "name": "chambered_council",
        "file": "chambered_council.phi",
        "semantic": "chambered — 3 on 432 Hz + 3 on 528 Hz",
        "expected_circuit": "TWO 3-qubit GHZ chains, 4 CX gates, shallower depth",
    },
]


def get_token() -> str:
    token = get_key("IBM_QUANTUM_TOKEN")
    if not token:
        raise KeyError("IBM_QUANTUM_TOKEN not found in vault")
    return token


def compile_to_qasm(phi_file: str) -> tuple[str, dict]:
    """Compile a .phi file to OpenQASM 3.0 using phic --target quantum."""
    result = subprocess.run(
        [str(PHIC), str(EXPERIMENT_DIR / phi_file), "--target", "quantum"],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"phic failed for {phi_file}:\n{result.stderr}")

    output = result.stdout
    # Extract QASM block
    qasm_lines = []
    in_qasm = False
    for line in output.split("\n"):
        if line.startswith("OPENQASM 3.0;"):
            in_qasm = True
        if in_qasm:
            qasm_lines.append(line)
            if line.startswith("    c[5] = measure") or line.startswith("    c[5] = measure q[5]"):
                # Keep going a bit more for remaining measurements
                pass
            if "════" in line:
                break

    qasm = "\n".join(qasm_lines).strip()
    if not qasm:
        raise RuntimeError(f"No QASM found in phic output for {phi_file}")

    # Extract coherence info
    coherence = {}
    for line in output.split("\n"):
        if ":" in line and "seat_" in line and not line.startswith("//"):
            parts = line.strip().split(":")
            if len(parts) == 2:
                try:
                    coherence[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    pass

    return qasm, coherence


def run_simulator(qasm: str, shots: int = 4096) -> dict:
    """Run QASM on the Qiskit Aer simulator. Returns {bitstring: probability}."""
    from qiskit.qasm3 import loads as qasm3_loads
    from qiskit_aer import AerSimulator

    circuit = qasm3_loads(qasm)
    sim = AerSimulator()
    job = sim.run(circuit, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Normalize to probabilities
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def submit_to_hardware(qasm: str, backend_name: str, shots: int) -> str:
    """Submit QASM to IBM Quantum hardware. Returns job_id."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.qasm3 import loads as qasm3_loads

    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    backend = service.backend(backend_name)

    circuit = qasm3_loads(qasm)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa_circuit = pm.run(circuit)

    print(f"  Transpiled: depth={isa_circuit.depth()}, ops={dict(isa_circuit.count_ops())}")

    sampler = SamplerV2(backend)
    job = sampler.run([(isa_circuit,)], shots=shots)
    return job.job_id()


def poll_hardware(job_id: str, backend_name: str) -> dict:
    """Poll a hardware job and return {bitstring: probability}."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = get_token()
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token)
    job = service.job(job_id)
    result = job.result()

    # Extract counts from the first (only) pub result
    pub_result = result[0]
    counts = pub_result.data.c.get_counts()

    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def classical_fidelity(sim_dist: dict, hw_dist: dict) -> float:
    """Compute classical fidelity (squared Bhattacharyya coefficient) between two distributions."""
    all_keys = set(sim_dist.keys()) | set(hw_dist.keys())
    bc = sum(math.sqrt(sim_dist.get(k, 0) * hw_dist.get(k, 0)) for k in all_keys)
    return bc ** 2


def hellinger_distance(sim_dist: dict, hw_dist: dict) -> float:
    """Compute Hellinger distance between two distributions (0=identical, 1=disjoint)."""
    all_keys = set(sim_dist.keys()) | set(hw_dist.keys())
    return math.sqrt(0.5 * sum((math.sqrt(sim_dist.get(k, 0)) - math.sqrt(hw_dist.get(k, 0))) ** 2 for k in all_keys))


def top_outcomes(dist: dict, n: int = 8) -> list:
    """Return the top N outcomes by probability."""
    return sorted(dist.items(), key=lambda x: -x[1])[:n]


def run_experiment(args):
    """Run the full experiment: compile, simulate, submit/poll, analyze."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []

    print(f"\n{'='*70}")
    print(f"  Semantic Coherence Experiment — {timestamp}")
    print(f"{'='*70}\n")

    # Step 1: Compile all programs
    print("Step 1: Compiling .phi programs to QASM...")
    compiled = {}
    for prog in PROGRAMS:
        name = prog["name"]
        print(f"  Compiling {name}...")
        qasm, coherence = compile_to_qasm(prog["file"])
        compiled[name] = {"qasm": qasm, "coherence": coherence, "meta": prog}
        print(f"    QASM: {len(qasm)} chars, coherence: {coherence}")

    # Step 2: Run simulator
    print(f"\nStep 2: Running Qiskit Aer simulator ({args.shots} shots)...")
    for name, data in compiled.items():
        print(f"  Simulating {name}...")
        sim_dist = run_simulator(data["qasm"], args.shots)
        data["sim_dist"] = sim_dist
        top = top_outcomes(sim_dist, 5)
        print(f"    Top outcomes: {top}")

    # Step 3: Submit to hardware or poll existing jobs
    if args.poll:
        print(f"\nStep 3: Polling hardware jobs: {args.poll}")
        job_ids = args.poll
        for i, name in enumerate(compiled.keys()):
            if i < len(job_ids):
                print(f"  Polling {name} (job {job_ids[i]})...")
                hw_dist = poll_hardware(job_ids[i], args.backend)
                compiled[name]["hw_dist"] = hw_dist
                compiled[name]["job_id"] = job_ids[i]
                top = top_outcomes(hw_dist, 5)
                print(f"    Top outcomes: {top}")

    elif args.backend and not args.sim_only:
        print(f"\nStep 3: Submitting to IBM Quantum ({args.backend}, {args.shots} shots)...")
        job_ids = []
        for name, data in compiled.items():
            print(f"  Submitting {name}...")
            job_id = submit_to_hardware(data["qasm"], args.backend, args.shots)
            data["job_id"] = job_id
            job_ids.append(job_id)
            print(f"    Job: {job_id}")

        print(f"\n  All jobs submitted. Poll later with:")
        print(f"  python3.12 scripts/semantic_coherence_experiment.py --backend {args.backend} --poll {' '.join(job_ids)}")
        print(f"\n  Or wait and poll now (jobs may take 10-60 minutes)...")

        if args.wait:
            import time
            for name, data in compiled.items():
                print(f"  Polling {name} (job {data['job_id']})...")
                while True:
                    try:
                        hw_dist = poll_hardware(data["job_id"], args.backend)
                        data["hw_dist"] = hw_dist
                        break
                    except Exception as e:
                        print(f"    Not ready yet: {e}")
                        time.sleep(30)

    # Step 4: Analyze
    has_hw = any("hw_dist" in d for d in compiled.values())
    if has_hw:
        print(f"\nStep 4: Analyzing simulator vs hardware fidelity...\n")
        print(f"{'Program':<25} {'Sim Entropy':>12} {'HW Entropy':>12} {'Fidelity':>10} {'Hellinger':>10}")
        print("-" * 75)

        for name, data in compiled.items():
            if "hw_dist" not in data:
                continue
            sim = data["sim_dist"]
            hw = data["hw_dist"]
            fid = classical_fidelity(sim, hw)
            hel = hellinger_distance(sim, hw)
            sim_ent = -sum(p * math.log2(p) for p in sim.values() if p > 0)
            hw_ent = -sum(p * math.log2(p) for p in hw.values() if p > 0)

            data["fidelity"] = fid
            data["hellinger"] = hel
            data["sim_entropy"] = sim_ent
            data["hw_entropy"] = hw_ent

            print(f"{name:<25} {sim_ent:>12.4f} {hw_ent:>12.4f} {fid:>10.4f} {hel:>10.4f}")

    # Step 5: Save report
    report = {
        "timestamp": timestamp,
        "shots": args.shots,
        "backend": args.backend or "simulator_only",
        "programs": {},
    }
    for name, data in compiled.items():
        report["programs"][name] = {
            "semantic": data["meta"]["semantic"],
            "expected_circuit": data["meta"]["expected_circuit"],
            "coherence": data["coherence"],
            "qasm": data["qasm"],
            "sim_dist": data.get("sim_dist"),
            "hw_dist": data.get("hw_dist"),
            "job_id": data.get("job_id"),
            "fidelity": data.get("fidelity"),
            "hellinger": data.get("hellinger"),
            "sim_entropy": data.get("sim_entropy"),
            "hw_entropy": data.get("hw_entropy"),
        }

    report_path = REPORTS_DIR / f"semantic_coherence_{datetime.now().strftime('%Y-%m-%d')}.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # Print QASM for inspection
    if args.show_qasm:
        for name, data in compiled.items():
            print(f"\n=== {name} QASM ===")
            print(data["qasm"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Semantic Coherence Experiment")
    parser.add_argument("--backend", default=None, help="IBM backend name (e.g., ibm_marrakesh)")
    parser.add_argument("--shots", type=int, default=4096, help="Number of shots")
    parser.add_argument("--sim-only", action="store_true", help="Only run simulator, no hardware")
    parser.add_argument("--poll", nargs='+', help="Poll existing job IDs")
    parser.add_argument("--wait", action="store_true", help="Wait for hardware jobs to complete")
    parser.add_argument("--show-qasm", action="store_true", help="Print QASM for each program")
    args = parser.parse_args()

    run_experiment(args)
