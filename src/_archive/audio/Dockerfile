# Quantum Audio Dockerfile (φ^φ)
FROM python:3.10-slim

# Set quantum environment
ENV PYTHONUNBUFFERED=1
ENV QUANTUM_FREQ=528
ENV PHI_RATIO=1.618033988749895
ENV SAMPLE_RATE=432000

# Install dependencies
RUN pip install --no-cache-dir \
    numpy \
    torch \
    sounddevice \
    scipy

# Create quantum workspace
WORKDIR /app
COPY audio /app/audio

# Start quantum service
CMD ["python", "-m", "audio.quantum_audio"]
