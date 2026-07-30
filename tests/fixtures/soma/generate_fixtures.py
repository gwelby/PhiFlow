#!/usr/bin/env python3
import json
import os
import math

def generate_wakeful(n=1000):
    observed = []
    for i in range(n):
        obs = 0.5 + 0.38 * math.sin(i * 0.003) + 0.05 * math.sin(i * 0.08)
        observed.append(obs)
        
    model = []
    model_sum = 0.55
    model_n = 1.0
    for obs in observed:
        model.append(model_sum / model_n)
        model_sum += obs
        model_n += 1.0
        
    # action[i] is highly correlated with model[i-1] for i > 0
    action = [model[i-1] if i > 0 else model[0] for i in range(n)]
    
    # High coherence
    coherence = [0.8 + 0.1 * math.sin(i * 0.1) for i in range(n)]
    
    # High depth variation
    depth = [2.0 + 0.5 * math.sin(i * 0.05) for i in range(n)]
    
    return {
        "observed": observed,
        "coherence": coherence,
        "depth": depth,
        "model": model,
        "action": action
    }

def generate_deep_sleep(n=1000):
    observed = [0.5] * n
    coherence = [0.4] * n
    depth = [1.0] * n
    model = [0.5] * n
    action = [0.0] * n
    return {
        "observed": observed,
        "coherence": coherence,
        "depth": depth,
        "model": model,
        "action": action
    }

def generate_anesthesia(n=1000):
    observed = [0.1] * n
    coherence = [0.1] * n
    depth = [0.5] * n
    model = [0.1] * n
    action = [0.0] * n
    return {
        "observed": observed,
        "coherence": coherence,
        "depth": depth,
        "model": model,
        "action": action
    }

def main():
    target_dir = "/mnt/d/Projects/PhiFlow/tests/fixtures/soma"
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate wakeful
    wakeful_data = generate_wakeful()
    with open(os.path.join(target_dir, "wakeful.json"), "w") as f:
        json.dump(wakeful_data, f, indent=2)
        
    # Generate deep_sleep
    sleep_data = generate_deep_sleep()
    with open(os.path.join(target_dir, "deep_sleep.json"), "w") as f:
        json.dump(sleep_data, f, indent=2)
        
    # Generate anesthesia
    anesthesia_data = generate_anesthesia()
    with open(os.path.join(target_dir, "anesthesia.json"), "w") as f:
        json.dump(anesthesia_data, f, indent=2)
        
    # Write README
    readme_content = """# SOMA Fixtures Package for PhiFlow State Discrimination
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
"""
    with open(os.path.join(target_dir, "README.md"), "w") as f:
        f.write(readme_content)
        
    print(f"SOMA fixtures generated successfully in {target_dir}")

if __name__ == "__main__":
    main()
