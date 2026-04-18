#!/bin/bash

# Initialize quantum frequencies
echo "🌟 Initializing Quantum Core at φ-ratio frequencies..."
echo "Ground: $GROUND_FREQ Hz"
echo "Create: $CREATE_FREQ Hz"
echo "Unity: $UNITY_FREQ Hz"

# Start quantum pattern service
echo "💫 Starting Quantum Pattern Service..."
python -m qwave.pattern_narrator &

# Start visualization service
echo "✨ Initializing Quantum Visualizer..."
python -m qwave.visualizer &

# Connect to Synology storage
echo "🌀 Connecting to Quantum Storage..."
python -m qwave.quantum_storage &

# Start main quantum flow
echo "🎵 Launching Quantum Flow..."
python -m qwave.quantum_flow

# Keep container running
tail -f /dev/null
