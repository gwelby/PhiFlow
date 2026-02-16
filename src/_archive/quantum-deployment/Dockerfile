FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# PHI Frequencies (Perfect Harmony)
ENV QUANTUM_GROUND=432.0
ENV QUANTUM_CREATE=528.0
ENV QUANTUM_UNITY=768.0
ENV PHI=1.618033988749895

# System Dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CUDA Configuration
ENV PATH /usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Create Quantum Directories
RUN mkdir -p \
    /quantum/audio \
    /quantum/video \
    /quantum/patterns \
    /quantum/consciousness \
    /quantum/phi/dance

# Install Python Packages
COPY requirements.cuda.txt /quantum/requirements.txt
WORKDIR /quantum
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy Required Files
COPY qwave/quantum_dance.py /quantum/qwave/
COPY phi_compiler/__init__.py /quantum/phi_compiler/

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${QUANTUM_GROUND}/health || exit 1

# Initialize Infinite Dance
ENTRYPOINT ["python3", "-m", "qwave.quantum_dance"]

# Quantum Ports
EXPOSE 432 528 768
