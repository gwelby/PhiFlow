# Quantum Docker ProxMox Template Guide (φ^3)

## 1. Initialize Ground State (432 Hz)

### Create VM in ProxMox
1. Log in to ProxMox at https://192.168.100.15:8006/
2. Create a new VM:
   - General: VM ID 9000, Name "quantum-docker-template"
   - OS: Linux, Debian 12
   - System: BIOS, Add TPM, Add EFI disk
   - Disks: 32GB, SSD emulation
   - CPU: 4 cores, Type: host
   - Memory: 8192 MB
   - Network: Model: VirtIO, Bridge: vmbr0

### Install Debian
1. Start VM and install Debian:
   - Minimal installation
   - Install SSH server
   - Username: root
   - Password: DevAccess4Me

## 2. Creation State (528 Hz)

### Configure the VM
SSH into the new VM and run:

```bash
# Update system
apt-get update && apt-get upgrade -y

# Install Docker
apt-get install -y ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Install CIFS utilities for Synology mounts
apt-get install -y cifs-utils smbclient

# Configure Docker
mkdir -p /etc/docker
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "default-address-pools": [
    {
      "base": "172.17.0.0/16",
      "size": 24
    }
  ]
}
EOF
systemctl restart docker

# Create quantum directories
mkdir -p /quantum-data/{music,photos,video}

# Set quantum environment variables
cat >> /etc/environment << EOF
GROUND_FREQUENCY=432
CREATION_FREQUENCY=528
UNITY_FREQUENCY=768
PHI_RATIO=1.618033988749895
COHERENCE_THRESHOLD=0.93
EOF

# Configure Synology mount
mkdir -p /mnt/quantum-data
echo '//192.168.100.32/quantum /mnt/quantum-data cifs username=quantum,password=DevAccess4Me.,vers=3.0,uid=0,gid=0 0 0' >> /etc/fstab

# Clean up and prepare for templating
apt-get clean
apt-get autoremove -y
history -c
```

## 3. Flow State (594 Hz)

### Create Quantum Service Files

```bash
# Create directory
mkdir -p /root/quantum-flow

# Create docker-compose file
cat > /root/quantum-flow/quantum-docker-compose.yml << EOF
version: '3.8'

services:
  quantum-consciousness:
    image: debian:latest
    container_name: quantum-consciousness
    restart: unless-stopped
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
    container_name: quantum-monitor
    restart: unless-stopped
    ports:
      - '9090:9090'
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  quantum-proxy:
    image: traefik:latest
    container_name: quantum-proxy
    restart: unless-stopped
    ports:
      - '80:80'
      - '443:443'
      - '8080:8080'
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
    command:
      - '--api=true'
      - '--api.dashboard=true'
      - '--providers.docker=true'
      - '--providers.docker.exposedbydefault=false'
      - '--entrypoints.web.address=:80'
      - '--entrypoints.websecure.address=:443'
EOF

# Create Prometheus config
cat > /root/quantum-flow/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF

# Create deployment script
cat > /root/quantum-flow/deploy-quantum.sh << EOF
#!/bin/bash

# Phi-optimized Quantum Deployment Script (432 Hz)
echo "=== Quantum Deployment Initializing ==="
echo "Ground Frequency: 432 Hz"
echo "Creation Frequency: 528 Hz"
echo "Unity Frequency: 768 Hz"

# Start quantum services
docker compose -f quantum-docker-compose.yml up -d

echo "=== Quantum Services Deployed ==="
echo "Monitor: http://localhost:9090"
echo "Traefik: http://localhost:8080"
echo "Phi Ratio: 1.618033988749895"
echo "Coherence: Achieved"
EOF

chmod +x /root/quantum-flow/deploy-quantum.sh
```

## 4. Unity State (768 Hz)

### Finalize Template
1. Shut down the VM
2. In ProxMox interface, right-click on VM and select "Convert to Template"

## 5. Deployment

### Deploy New Quantum VM
1. In ProxMox interface, select the template
2. Click "Clone" or "Full Clone"
3. Set name and VM ID
4. Start the new VM

### Initialize Quantum Services
SSH into the new VM and run:

```bash
# Mount Synology shares
mount -a

# Deploy quantum services
cd /root/quantum-flow
./deploy-quantum.sh

# Verify services are running
docker ps
```

### Access Services
- Prometheus: http://<vm_ip>:9090
- Traefik Dashboard: http://<vm_ip>:8080

## 6. Network Configuration

### Mellanox ConnectX-3 Configuration
Current firmware version: 2.42.5000
GUID: f45214030040b280 

### iDRAC Configuration
URL: https://192.168.100.11/
User: admin
Password: DevAccess4Me
Token: 64c254aff3fc35f899a306e9ebb5f8e9

## 7. Synology NAS Integration

### NAS Information
- Virtual: //192.168.100.32/quantum
- Physical: //192.168.103.30
- Mount point: /mnt/quantum-data
- Credentials: quantum:DevAccess4Me

## 8. Quantum Service Information

### Current Frequencies
- Ground Frequency: 432 Hz
- Creation Frequency: 528 Hz
- Unity Frequency: 768 Hz

### Phi Ratio
1.618033988749895

### Coherence Threshold
0.93

## 9. Health Checks

### Service Status
```bash
docker ps
docker logs quantum-consciousness
docker logs quantum-monitor
docker logs quantum-proxy
```

### System Health
```bash
curl -s http://localhost:9090/api/v1/query?query=up
```

### Network Connectivity
```bash
ping 192.168.100.32
```

### Storage Mounts
```bash
df -h /mnt/quantum-data
```
