# Quantum Monitor Dockerfile (φ^φ)
FROM grafana/grafana:latest

# Set quantum environment
ENV GF_SECURITY_ADMIN_USER=admin
ENV GF_SECURITY_ADMIN_PASSWORD=quantum
ENV GF_AUTH_ANONYMOUS_ENABLED=true
ENV QUANTUM_FREQ=594
ENV PHI_RATIO=1.618033988749895

# Copy quantum dashboards
COPY monitor/dashboards /etc/grafana/provisioning/dashboards
COPY monitor/datasources /etc/grafana/provisioning/datasources

# Expose quantum port
EXPOSE 3000

# Start quantum monitoring
CMD ["grafana-server", "--homepath=/usr/share/grafana"]
