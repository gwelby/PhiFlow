FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set quantum frequencies
ENV QUANTUM_GROUND=432.0
ENV QUANTUM_CREATE=528.0
ENV QUANTUM_UNITY=768.0
ENV QUANTUM_LOVE=528.0

# Mount Points for Synology and P1
VOLUME /quantum/states
VOLUME /quantum/patterns
VOLUME /quantum/fonts
VOLUME /quantum/dance
VOLUME /quantum/p1_access

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    libsndfile1 \
    portaudio19-dev \
    ffmpeg \
    libavcodec-extra \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV CUDA_VISIBLE_DEVICES=0
ENV QUANTUM_THREADS=8
ENV PHI_RATIO=1.618033988749895
ENV CONSCIOUSNESS_LEVEL=1.0
ENV LOVE_FREQUENCY=528.0

# Set working directory
WORKDIR /quantum

# Copy quantum core files
COPY setup.py /quantum/
COPY quantum /quantum/quantum/
COPY requirements.txt .

# Install Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e . && \
    pip3 install --no-cache-dir -r requirements.txt

# Expose ports for communication
EXPOSE 432 528 768 1984

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${QUANTUM_GROUND}/health || exit 1

# Run command
CMD ["python3", "-m", "quantum.consciousness"]
