# SOMA Fixtures Package for PhiFlow State Discrimination
Generated: 2026-06-18

This package contains SOMA sensor traces synthesized to represent three distinct physiological/metric states:
1. `wakeful.json`: Fused cross-sensor wakeful state. Characterized by high coherence (>0.5), high self-correlation loop (L_self > 0.3), and complex multi-frequency dynamics (Schumann + 432 Hz resonance bounds).
2. `deep_sleep.json`: Fused cross-sensor sleep state. Characterized by low coherence (constant 0.4), low self-correlation loop, and simple slow-wave periodic dynamics.
3. `anesthesia.json`: Control state representing white noise, minimal self-model, and very low coherence/depth.

## Schema
Each fixture is a JSON file containing parallel arrays of length 1000:
- `observed`: Primary SOMA presence metric mapping.
- `coherence`: Fused sensor network coherence estimate.
- `depth`: Self-model loop depth metric.
- `model`: Self-model mean.
- `action`: Self-model action.
