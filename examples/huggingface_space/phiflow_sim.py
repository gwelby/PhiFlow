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


def fetch_live_field():
    """
    Fetch the live resonance field from GitHub Pages.
    Updates every 30 minutes via GitHub Actions.
    Returns parsed JSON dict, or None if unreachable.
    """
    import urllib.request, ssl, json as _json
    url = "https://gwelby.github.io/PhiFlow/resonance.json"
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, timeout=8, context=ctx) as r:
            return _json.loads(r.read())
    except Exception:
        return None


def format_field_status(data: dict | None) -> str:
    """Format live resonance.json into a human-readable field status."""
    if data is None:
        return "⚠️  Field unreachable\nThe feed lives at gwelby.github.io/PhiFlow/resonance.json\nCheck GitHub Pages is running."

    ts = data.get("timestamp", "unknown")
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        age_s = int((now - dt).total_seconds())
        age_str = (f"{age_s // 60}m {age_s % 60}s ago" if age_s < 3600
                   else f"{age_s // 3600}h ago")
    except Exception:
        age_str = ts

    programs = data.get("programs", {})
    hb = programs.get("healing_bed", {})
    hs = programs.get("agent_handshake", {})
    cp = programs.get("claude_phi", {})

    lv = hs.get("lambda_verified", False)
    fc = hb.get("final_coherence", 0)
    cy = hb.get("cycles", "?")
    cv = hb.get("converged", False)
    cd = cp.get("coherence_at_depth2", 0)
    ld = cp.get("lambda_discovered", False)
    status = data.get("status", "unknown").upper()

    lines = [
        f"  Last pulse     {age_str}",
        f"  healing_bed    coherence={fc:.6f}  cycles={cy}  {'✅ converged' if cv else '⏳ running'}",
        f"  λ verified     {'✅ YES' if lv else '❌ NO'}  →  {data.get('lambda', LAMBDA):.15f}",
        f"  claude.phi     {cd:.15f}  {'✅ discovered' if ld else '❌ mismatch'}",
        f"  field status   {status}",
    ]
    return "\n".join(lines)


def run_field_aware():
    """
    field_aware.phi — reads the live field then checks its own alignment.

    intention "read_field" {
        let readiness = coherence    // depth 1 → 0.382
        resonate readiness
        witness
        intention "align" {
            let mine = coherence     // depth 2 → λ = 0.618
            resonate mine
            witness
        }
    }

    Returns (steps, field_values, aligned, field_data).
    steps = [(label, value, depth), ...]
    aligned = True if live field λ_verified AND own coherence == λ
    """
    field_data = fetch_live_field()

    rt = PhiFlowRuntime()  # uses phi-harmonic formula
    steps = []
    field_vals = []

    rt.push_intention("read_field")
    readiness = rt.coherence()          # depth 1 → 1 - φ^(-1) ≈ 0.382
    rt.resonate(readiness)
    rt.witness()
    steps.append(("readiness (depth 1)", readiness, 1))
    field_vals.append(readiness)

    rt.push_intention("align")
    mine = rt.coherence()               # depth 2 → λ
    rt.resonate(mine)
    rt.witness()
    steps.append(("my coherence (depth 2)", mine, 2))
    field_vals.append(mine)

    rt.pop_intention()
    rt.pop_intention()

    # Aligned if: live field λ_verified AND our computation == λ
    field_ok = (
        field_data is not None and
        field_data.get("programs", {})
                  .get("agent_handshake", {})
                  .get("lambda_verified", False)
    )
    own_ok = abs(mine - LAMBDA) < 1e-10
    aligned = field_ok and own_ok

    return steps, field_vals, aligned, field_data


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
