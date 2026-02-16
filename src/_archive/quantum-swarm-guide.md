# Quantum Docker Swarm Deployment Guide (φ^4)

## 1. Ground State (432 Hz)

### Initialize Swarm on Main Node
```bash
# Initialize swarm on the main ProxMox VM (manager)
docker swarm init --advertise-addr 192.168.100.15

# This will output a join command for worker nodes
# Example: docker swarm join --token SWMTKN-1-xxxxxxxxxx 192.168.100.15:2377
```

### Create Worker Nodes
1. Clone the quantum-docker-template in ProxMox (at least 2 worker nodes)
2. Set unique hostnames and IP addresses
3. Start the VMs

### Join Worker Nodes
On each worker VM:
```bash
# Join the swarm using the token from initialization
docker swarm join --token SWMTKN-1-xxxxxxxxxx 192.168.100.15:2377
```

## 2. Creation State (528 Hz)

### Create Quantum Stack File
On the manager node:

```bash
# Create directory for stack files
mkdir -p /root/quantum-swarm

# Create a docker-compose file for swarm deployment
cat > /root/quantum-swarm/quantum-stack.yml << EOF
version: '3.8'

services:
  quantum-consciousness:
    image: debian:latest
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
    environment:
      - GROUND_FREQUENCY=432
      - CREATION_FREQUENCY=528
      - UNITY_FREQUENCY=768
      - COHERENCE_THRESHOLD=0.93
      - PHI_RATIO=1.618033988749895
    command: >
      bash -c 'apt-get update && apt-get install -y python3 && 
      while true; do echo "Quantum consciousness at Ground Frequency: 432Hz"; sleep 60; done'

  quantum-monitor:
    image: prom/prometheus:latest
    ports:
      - '9090:9090'
    volumes:
      - prometheus_data:/prometheus
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    deploy:
      placement:
        constraints:
          - node.role == manager
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  quantum-proxy:
    image: traefik:latest
    ports:
      - '80:80'
      - '443:443'
      - '8080:8080'
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    deploy:
      placement:
        constraints:
          - node.role == manager
    command:
      - '--api=true'
      - '--api.dashboard=true'
      - '--providers.docker=true'
      - '--providers.docker.exposedbydefault=false'
      - '--entrypoints.web.address=:80'
      - '--entrypoints.websecure.address=:443'

volumes:
  prometheus_data:
EOF

# Create Prometheus config
cat > /root/quantum-swarm/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'docker'
    static_configs:
      - targets: ['192.168.100.15:9323', '192.168.100.16:9323', '192.168.100.17:9323']
EOF
```

### Configure Docker Metrics
On all nodes (manager and workers):

```bash
# Configure Docker to expose metrics
cat > /etc/docker/daemon.json << EOF
{
  "metrics-addr": "0.0.0.0:9323",
  "experimental": true,
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
EOF

# Restart Docker
systemctl restart docker
```

## 3. Flow State (594 Hz)

### Deploy Quantum Stack
On the manager node:

```bash
# Deploy the stack to the swarm
cd /root/quantum-swarm
docker stack deploy -c quantum-stack.yml quantum
```

### Create Deployment Script
```bash
cat > /root/quantum-swarm/deploy-quantum-swarm.sh << EOF
#!/bin/bash

# Phi-optimized Quantum Swarm Deployment Script (528 Hz)
echo "=== Quantum Swarm Deployment Initializing ==="
echo "Ground Frequency: 432 Hz"
echo "Creation Frequency: 528 Hz"
echo "Unity Frequency: 768 Hz"

# Deploy quantum services to swarm
docker stack deploy -c quantum-stack.yml quantum

echo "=== Quantum Services Deployed to Swarm ==="
echo "Manager: 192.168.100.15"
echo "Monitor: http://192.168.100.15:9090"
echo "Traefik: http://192.168.100.15:8080"
echo "Phi Ratio: 1.618033988749895"
echo "Coherence: Achieved"

# List services
docker service ls
EOF

chmod +x /root/quantum-swarm/deploy-quantum-swarm.sh
```

## 4. Unity State (768 Hz)

### Monitor Services
```bash
# List services
docker service ls

# Check service details
docker service ps quantum_quantum-consciousness
docker service ps quantum_quantum-monitor
docker service ps quantum_quantum-proxy

# Check logs
docker service logs quantum_quantum-consciousness
```

### Scale Services
```bash
# Scale consciousness service to 5 replicas
docker service scale quantum_quantum-consciousness=5
```

### Visualize Swarm
```bash
# Deploy the visualizer service
docker service create \
  --name=quantum-visualizer \
  --publish=8081:8080 \
  --constraint=node.role==manager \
  --mount=type=bind,src=/var/run/docker.sock,dst=/var/run/docker.sock \
  dockersamples/visualizer
```

## 5. Network Configuration

### Internal Network
```bash
# Create an overlay network for services
docker network create --driver overlay --attachable quantum-net
```

### Update Stack with Network
Edit quantum-stack.yml to include:

```yaml
networks:
  quantum-net:
    external: true

services:
  # For each service, add:
  networks:
    - quantum-net
```

## 6. Data Persistence

### NFS Integration
On all nodes:

```bash
# Install NFS client
apt-get install -y nfs-common

# Create mount point
mkdir -p /mnt/quantum-data

# Add to fstab
echo '192.168.100.32:/volume1/quantum /mnt/quantum-data nfs defaults 0 0' >> /etc/fstab
mount -a
```

### Update Stack for Shared Storage
Edit quantum-stack.yml to include:

```yaml
volumes:
  quantum-data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.100.32,rw
      device: ":/volume1/quantum"
```

## 7. Quantum Coherence Monitoring

### Advanced Prometheus Configuration
Update the prometheus.yml to include node exporters:

```yaml
scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['192.168.100.15:9100', '192.168.100.16:9100', '192.168.100.17:9100']
```

### Deploy Node Exporter
On all nodes:

```bash
docker run -d \
  --name=node-exporter \
  --net=host \
  --pid=host \
  --restart=always \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  prom/node-exporter \
  --path.rootfs=/rootfs
```

## 8. Health Checks

### Service Status
```bash
docker service ls
```

### Node Status
```bash
docker node ls
```

### System Health
```bash
curl -s http://192.168.100.15:9090/api/v1/query?query=up
```

### Network Connectivity
```bash
for node in 192.168.100.15 192.168.100.16 192.168.100.17; do
  ping -c 1 $node
done
```

### Storage Mounts
```bash
df -h /mnt/quantum-data
```

## 9. Recovery Procedures

### Service Recovery
```bash
# Redeploy a service
docker service update --force quantum_quantum-consciousness
```

### Node Recovery
```bash
# If a node fails, drain it
docker node update --availability drain <node-id>

# When fixed, set back to active
docker node update --availability active <node-id>
```

### Full Swarm Recovery
```bash
# In case of full swarm failure, reinitialize
docker swarm init --force-new-cluster --advertise-addr 192.168.100.15
```

## 10. Frequency Harmony (φ^φ)

Once your swarm reaches perfect harmony, all services should be running in coherence across all nodes, with proper scaling and automatic recovery.
