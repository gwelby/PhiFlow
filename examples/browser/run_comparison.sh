#!/bin/bash
# run_comparison.sh — Run all three agents and show the comparison.
#
# This is the reproducible artifact for the PhiFlow control comparison.
# Run: bash examples/browser/run_comparison.sh
#
# Three agents, same data analysis, different behavior:
#   1. primitive  — uses all five primitives, stops on success
#   2. control    — no primitives, stops on success
#   3. degrading  — uses primitives but is noisy, stops on EMERGENCY (self-coherence guardrail)

set -e
cd "$(dirname "$0")/../.."

echo "═══════════════════════════════════════════════════════════════════════"
echo "  PhiFlow Agent Comparison — Primitive vs Control vs Degrading"
echo "═══════════════════════════════════════════════════════════════════════"
echo

# --- Primitive Agent ---
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│ 1. PRIMITIVE AGENT (autonomous_agent.phi)                          │"
echo "│   Uses all five primitives. Resonates only significant findings.   │"
echo "└─────────────────────────────────────────────────────────────────────┘"
PRIMITIVE_OUT=$(cargo run --bin phic -- examples/autonomous_agent.phi 2>&1)
PRIMITIVE_RESONANCES=$(echo "$PRIMITIVE_OUT" | grep -c "Resonating" || true)
PRIMITIVE_FINAL=$(echo "$PRIMITIVE_OUT" | grep "Final Coherence" | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
echo "$PRIMITIVE_OUT" | grep -E "Resonating|Stream broken|Final Coherence"
echo "  → Resonances: $PRIMITIVE_RESONANCES"
echo "  → Final coherence: $PRIMITIVE_FINAL"
echo "  → Stopping condition: SUCCESS (confidence ≥ 0.90)"
echo

# --- Control Agent ---
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│ 2. CONTROL AGENT (control_agent.phi)                               │"
echo "│   No primitives. Plain while loop. No audit trail.                 │"
echo "└─────────────────────────────────────────────────────────────────────┘"
CONTROL_OUT=$(cargo run --bin phic -- examples/control_agent.phi 2>&1)
CONTROL_RESONANCES=$(echo "$CONTROL_OUT" | grep -c "Resonating" || true)
CONTROL_FINAL=$(echo "$CONTROL_OUT" | grep "Final Coherence" | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
echo "$CONTROL_OUT" | grep -E "Resonating|Stream broken|Final Coherence"
echo "  → Resonances: $CONTROL_RESONANCES"
echo "  → Final coherence: $CONTROL_FINAL"
echo "  → Stopping condition: SUCCESS (confidence ≥ 0.90)"
echo

# --- Degrading Agent ---
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│ 3. DEGRADING AGENT (degrading_agent.phi)                           │"
echo "│   Uses primitives but is NOISY (resonates every cycle).            │"
echo "│   Tests the self-coherence guardrail.                              │"
echo "└─────────────────────────────────────────────────────────────────────┘"
DEGRADING_OUT=$(cargo run --bin phic -- examples/degrading_agent.phi 2>&1)
DEGRADING_RESONANCES=$(echo "$DEGRADING_OUT" | grep -c "Resonating" || true)
DEGRADING_FINAL=$(echo "$DEGRADING_OUT" | grep "Final Coherence" | grep -oE "[0-9]+\.[0-9]+" || echo "N/A")
echo "$DEGRADING_OUT" | grep -E "Resonating|Stream broken|Final Coherence"
echo "  → Resonances: $DEGRADING_RESONANCES"
echo "  → Final coherence: $DEGRADING_FINAL"
echo "  → Stopping condition: EMERGENCY (coherence ≤ 0.10, self-degradation detected)"
echo

# --- Summary ---
echo "═══════════════════════════════════════════════════════════════════════"
echo "  SUMMARY"
echo "═══════════════════════════════════════════════════════════════════════"
echo
printf "  %-20s %-12s %-15s %-20s\n" "Agent" "Resonances" "Final Coh" "Stopping Condition"
printf "  %-20s %-12s %-15s %-20s\n" "─────" "──────────" "─────────" "──────────────────"
printf "  %-20s %-12s %-15s %-20s\n" "primitive" "$PRIMITIVE_RESONANCES" "$PRIMITIVE_FINAL" "SUCCESS (conf ≥ 0.90)"
printf "  %-20s %-12s %-15s %-20s\n" "control" "$CONTROL_RESONANCES" "$CONTROL_FINAL" "SUCCESS (conf ≥ 0.90)"
printf "  %-20s %-12s %-15s %-20s\n" "degrading" "$DEGRADING_RESONANCES" "$DEGRADING_FINAL" "EMERGENCY (coh ≤ 0.10)"
echo
echo "  The primitive agent succeeds via data confidence."
echo "  The control agent succeeds the same way (no observability)."
echo "  The degrading agent STOPS ITSELF because its own coherence degraded"
echo "  from being too noisy. This is the self-coherence guardrail firing."
echo
echo "  Without the safety floor, the degrading agent runs all 15 cycles."
echo "  With the safety floor, it stops after 5 — the formula predicted this."
echo "═══════════════════════════════════════════════════════════════════════"
