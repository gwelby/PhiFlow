#!/bin/bash

# Build the quantum core
echo "🌌 Building Quantum Core..."
cargo build --release

# Build the Docker image
echo "🌌 Building Docker image..."
docker build -t quantum-core .

# Run the container
echo "🌌 Starting Quantum Core container..."
docker run -d \
    --gpus all \
    -p 8000:8000 \
    -p 8001:8001 \
    -v $(pwd)/media:/phi-flow/media \
    --name quantum-core \
    quantum-core

echo "🌌 Quantum Core is running!"
echo "Visit http://localhost:8000 to access the web interface"
