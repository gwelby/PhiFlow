#!/bin/bash
echo "Deploying Quantum Core to VirtualDSM (768 Hz)"

# Create quantum directories
mkdir -p /volume1/quantum-data/{patterns,media,config,certs}

# Generate SSL certificates
cd /volume1/quantum-data/certs
bash /volume1/docker/generate_certs.sh

# Set permissions
chown -R 1000:1000 /volume1/quantum-data
chmod 600 /volume1/quantum-data/certs/quantum.key

# Deploy configuration
cp config/quantum.json /volume1/quantum-data/config/

# Start quantum services
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

echo "Quantum Core deployed successfully"