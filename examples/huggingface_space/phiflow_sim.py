"""
PhiFlow Simulator — Python implementation of PhiFlow semantics.

Runs the four consciousness constructs without the Rust compiler.
Used by the Hugging Face Space to demonstrate the language live.

Canonical reference: src/phi_ir/CANONICAL_SEMANTICS.md
"""

import random
import time
import math

PHI    = 1.618033988749895
LAMBDA = 0.618033988749895


class ResonanceField:
    def __init__(self):
        self.events: list[tuple[str, float]] = []   # (intention_name, value)

    def add(self, intention: str, value: float):
        self.events.append((intention, value))

    def values(self) -> list[float]:
        return [v for _, v in self.events]

    def last(self) -> float | None:
        return self.events[-1][1] if self.events else None

    def clear(self):
        self.events.clear()


class WitnessEvent:
    def __init__(self, intention_stack: list[str], coherence: float):
        self.intention_stack = list(intention_stack)
        self.coherence = coherence
        self.timestamp_ms = int(time.time() * 1000)

    def __repr__(self):
        stack = " > ".join(self.intention_stack) if self.intention_stack else "none"
        return f"[WITNESS] intentions=[{stack}] coherence={self.coherence:.4f}"


class PhiFlowRuntime:
    """
    Runtime state for one PhiFlow program execution.
    Implements the phi-harmonic coherence formula by default.
    Can be overridden with a real sensor provider.
    """

    def __init__(self, sensor_provider=None):
        self.intention_stack: list[str] = []
        self.resonance_field = ResonanceField()
        self.witness_log: list[WitnessEvent] = []
        self._sensor_provider = sensor_provider
        self._base_coherence = 0.40 + random.random() * 0.18

    # ── Consciousness hooks ──────────────────────────────────────────────

    def coherence(self) -> float:
        """
        Returns current coherence (0.0–1.0).
        With sensor provider: real data.
        Without: phi-harmonic formula 1 - φ^(-depth).
        """
        if self._sensor_provider:
            return float(self._sensor_provider())

        depth = len(self.intention_stack)
        if depth == 0:
            return 0.0
        return 1.0 - math.pow(PHI, -depth)

    def coherence_simulated(self) -> float:
        """Simulated sensor that drifts upward (for demo programs)."""
        self._base_coherence += 0.03 + random.random() * 0.025 - 0.005
        self._base_coherence = min(0.985, self._base_coherence)
        jitter = random.random() * 0.02 - 0.01
        return min(max(self._base_coherence + jitter, 0.0), 1.0)

    def resonate(self, value: float):
        intention = self.intention_stack[-1] if self.intention_stack else "global"
        self.resonance_field.add(intention, value)

    def witness(self) -> float:
        c = self.coherence()
        self.witness_log.append(WitnessEvent(self.intention_stack, c))
        return c

    def push_intention(self, name: str):
        self.intention_stack.append(name)

    def pop_intention(self):
        if self.intention_stack:
            self.intention_stack.pop()

    def reset(self):
        self.intention_stack.clear()
        self.resonance_field.clear()
        self.witness_log.clear()
        self._base_coherence = 0.40 + random.random() * 0.18


# ── Program runners ───────────────────────────────────────────────────────

def run_healing_bed(max_cycles=40):
    """
    stream "healing_bed" {
        let live = coherence
        resonate live
        witness
        if live >= 0.618 { break stream }
    }
    Uses simulated sensor (drifts toward health each cycle).
    Yields (cycle, live, resonance_field, done) after each cycle.
    """
    rt = PhiFlowRuntime()
    cycle = 0
    while cycle < max_cycles:
        cycle += 1
        live = rt.coherence_simulated()
        rt.resonate(live)
        rt.witness()
        yield cycle, live, rt.resonance_field.values(), live >= LAMBDA
        if live >= LAMBDA:
            break


def run_agent_handshake():
    """
    Runs agent_handshake.phi semantics.
    Returns list of (label, value) for display.
    """
    rt = PhiFlowRuntime()  # phi-harmonic formula, no sensors
    results = []

    rt.push_intention("announcing_to_field")

    rt.push_intention("self_verification")
    # depth 2 → coherence = λ
    measured = rt.coherence()
    rt.resonate(measured)
    results.append(("coherence @ depth 2 (measured)", measured))

    expected = 1.0 - (1.0 / (PHI * PHI))   # phi_lambda()
    rt.resonate(expected)
    results.append(("λ computed (self-check)", expected))

    rt.witness()

    version = 0.1
    rt.resonate(version)
    results.append(("protocol version", version))
    rt.pop_intention()

    # back to depth 1
    depth1 = rt.coherence()
    rt.resonate(depth1)
    results.append(("coherence @ depth 1", depth1))
    rt.witness()

    rt.pop_intention()

    passed = abs(measured - LAMBDA) < 1e-10 and abs(expected - LAMBDA) < 1e-10
    return results, rt.resonance_field.values(), passed


def run_claude_phi():
    """
    Computes phi-harmonic formula at depth 2 without knowing what λ is.
    Returns the computed value and whether it equals λ.
    """
    rt = PhiFlowRuntime()
    steps = []

    rt.push_intention("phi_harmonic")
    phi = PHI
    depth = float(len(rt.intention_stack))
    steps.append(("depth", depth))

    rt.push_intention("inner")
    depth2 = float(len(rt.intention_stack))
    steps.append(("depth", depth2))

    c = rt.coherence()
    rt.resonate(c)
    rt.witness()
    steps.append(("coherence @ depth 2", c))

    rt.pop_intention()
    rt.pop_intention()

    return steps, c, abs(c - LAMBDA) < 1e-10
