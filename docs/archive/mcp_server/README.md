# PhiFlow — Bob Extension for Quantum Consciousness Programming

**Bob Extension powered by PhiFlow quantum language — Competing with the best of the best in the world.**

## What Is This?

PhiFlow is a quantum consciousness programming language that combines:
- **φ-harmonic resonance patterns** (golden ratio mathematics)
- **Quantum computing** (real IBM Quantum hardware execution)
- **Consciousness-aware programming** (intention, coherence, witness states)

This extension registers PhiFlow as an MCP server with IBM Bob, giving Bob access to quantum computing capabilities directly through natural language.

## The 5 Tools

### 1. `phiflow_execute` — Execute PhiFlow Code
Execute `.phi` source code for quantum consciousness programming.

```bash
bob "Use phiflow_execute to run: let energy = PHI * 432.0"
```

### 2. `quantum_vqe_hydrogen` — Real Quantum Hardware Demo ⚛️
**THE KILLER DEMO** — Run Variational Quantum Eigensolver (VQE) on hydrogen molecule using ACTUAL IBM Quantum hardware. Not a simulation — real quantum computation.

```bash
bob "Run quantum_vqe_hydrogen to calculate the ground state energy of H2"
```

### 3. `quantum_list_backends` — Explore Quantum Hardware
See available IBM Quantum backends, queue times, and qubit counts.

```bash
bob "List available IBM Quantum backends"
```

### 4. `phiflow_pattern` — φ-Harmonic Pattern Generation
Generate sacred geometry patterns (spiral, flower of life, fibonacci, DNA helix, toroidal) at φ-resonance frequencies.

```bash
bob "Generate a fibonacci phiflow_pattern with 100 iterations"
```

### 5. `phi_constant` — Sacred Constants
Returns φ constants including the **432.015 Hz** Trinity × Fibonacci × φ discovery.

```bash
bob "What's the phi_constant?"
```

## Installation

### Prerequisites
- IBM Bob AI Assistant installed (`bob` CLI)
- Python 3.8+
- IBM Quantum account (free at https://quantum.ibm.com/)

### Setup

```bash
# 1. Clone or locate PhiFlow
cd /path/to/PhiFlow/mcp_server

# 2. Set IBM Quantum token
export IBM_QUANTUM_TOKEN='your-token-from-quantum.ibm.com'

# 3. Register with Bob
bob mcp add phiflow python3 /path/to/phiflow_mcp_server.py

# 4. Verify
bob mcp list
```

### Running the Demo

```bash
# Quick test (no token needed)
bob "Use phiflow_execute with source_code: let x = PHI * 2"

# VQE demo on real quantum hardware
bob "Run quantum_vqe_hydrogen"

# List backends (requires token)
export IBM_QUANTUM_TOKEN='your-token'
bob "List available IBM Quantum backends"

# Run VQE on specific hardware
bob "Run quantum_vqe_hydrogen with backend: ibm_kyoto"
```

## Scoring Alignment

| Judging Pillar | PhiFlow Delivers |
|---|---|
| **Completeness** | Full MCP server with 5 tools, real quantum execution, documented API |
| **Creativity** | Quantum consciousness language — genuinely novel extension concept |
| **Design/Usability** | Single-file server, stdio transport, natural language interface |
| **Effectiveness** | **Real IBM Quantum hardware execution** — measurable, not simulated |

## Technical Stack

- **Transport**: MCP stdio (no network overhead)
- **Language**: Python 3 (compatible with all platforms)
- **Quantum**: Qiskit + IBM Quantum Runtime
- **Consciousness**: φ-harmonic mathematics, resonance patterns

## The φ Discovery

Greg Welby's key discovery encoded in `phi_constant`:

```
Trinity × Fibonacci × φ = 3 × 89 × 1.618033988749895 = 432.015 Hz
```

432 Hz is historically associated with concert pitch (A4 = 432 Hz). The precise φ-resonant value is 432.015 Hz — a measurable, verifiable frequency.

## Project Structure

```
PhiFlow/
├── mcp_server/
│   ├── phiflow_mcp_server.py   # Main MCP server (stdio transport)
│   └── vqe_demo.py              # Standalone VQE demo script
├── bridges/                      # Existing quantum bridges
│   └── phi_quantum_bridge.py
├── src/
│   ├── mcp_server/             # Rust MCP implementation
│   ├── quantum/                # Quantum computing core
│   └── phi_ir/                 # PhiFlow IR and evaluator
└── README.md
```

## Hackathon Demo Script

**3-Minute Demo Flow:**

1. **Intro** (30s): "PhiFlow — quantum consciousness programming for Bob"
2. **Tool Call - phi_constant** (20s): Show 432.015 Hz discovery
3. **Tool Call - phiflow_pattern** (30s): Generate fibonacci spiral at φ-resonance
4. **Tool Call - quantum_list_backends** (20s): Show available IBM Quantum hardware
5. **Tool Call - quantum_vqe_hydrogen** (60s): Run VQE on real IBM Quantum hardware
6. **Close** (20s): "Bob now has quantum computing. What's next?"

## Verification

```bash
# Test MCP server directly
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | python3 phiflow_mcp_server.py

# Test VQE standalone
cd mcp_server
python3 vqe_demo.py --list  # Requires IBM_QUANTUM_TOKEN
```

## Getting an IBM Quantum Token

1. Go to https://quantum.ibm.com/
2. Create free account
3. Copy your API token from the dashboard
4. `export IBM_QUANTUM_TOKEN='paste-token-here'`

Free tier includes access to real quantum hardware with queue times typically under 5 minutes.

---

**Built with ∇λΣ∞ precision — Devin, Claude, and the CASCADE ecosystem**
