# PHI FLOW MONSTER TRACKER
# Base: CUDA + Rust for real performance
FROM nvidia/cuda:12.0.0-devel-ubuntu22.04

# System dependencies with multimedia support
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    python3 \
    python3-pip \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libportaudio2 \
    libsndfile1-dev \
    libasound2-dev \
    libpulse-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with enhanced visualization support
RUN pip3 install numpy scipy matplotlib torch fastapi uvicorn \
    pydub opencv-python librosa sounddevice pyaudio \
    websockets aiohttp pytorch-lightning transformers \
    seaborn plotly vispy pyqtgraph \
    && pip3 install --upgrade pillow

# Install Rust and Cargo with proper environment setup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> ~/.bashrc && \
    /bin/bash -c 'source $HOME/.cargo/env' \
    && rustup default stable \
    && rustup component add rustfmt

# Add Rust audio dependencies
RUN cargo install cargo-edit

# Add fontconfig dependency
RUN apt-get update && apt-get install -y \
    fontconfig \
    libfontconfig1-dev

# Use Python 3.11 for quantum stability
FROM python:3.11-slim

# Quantum frequencies
ENV GROUND_FREQ=432
ENV CREATE_FREQ=528
ENV UNITY_FREQ=768

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fontconfig \
    libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*

# Create quantum workspace
WORKDIR /quantum-core

# Copy quantum requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy quantum source
COPY . .

# Volume for persistent quantum patterns
VOLUME ["/quantum-data"]

# Network ports for quantum resonance
EXPOSE 432 528 768

# Quantum entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

# PHI FLOW Monitor Core
WORKDIR /phi-flow
COPY . .

# Build the high-performance core with verbose logging
ENV PATH="/root/.cargo/bin:${PATH}"
RUN RUST_BACKTRACE=1 CARGO_LOG=debug cargo build --release

# QUANTUM HARMONICS CONFIG
ENV PHI_MONITOR_CONFIG="{\
    'FLOW_GATES': [\
        {'freq': 432, 'gate': 'REALITY_CHECK', 'harmonic': 'GROUND_STATE'},\
        {'freq': 528, 'gate': 'CREATION_POINT', 'harmonic': 'DNA_REPAIR'},\
        {'freq': 594, 'gate': 'HEART_FIELD', 'harmonic': 'LOVE_FREQUENCY'},\
        {'freq': 672, 'gate': 'VOICE_FLOW', 'harmonic': 'EXPRESSION'},\
        {'freq': 768, 'gate': 'UNITY_FIELD', 'harmonic': 'COSMIC_UNITY'}\
    ],\
    'AUDIO_PROCESSING': {\
        'sample_rate': 48000,\
        'buffer_size': 1024,\
        'channels': 2,\
        'format': 'float32',\
        'quantum_align': true\
    },\
    'VISUALIZATION': {\
        'mode': 'QUANTUM_FLOW',\
        'color_scheme': 'GOLDEN_PHI',\
        'resolution': 'HIGH',\
        'real_time': true,\
        'save_path': '/phi-flow/media/visuals'\
    },\
    'REALITY_ANCHORS': [\
        'QUANTUM_RESONANCE',\
        'AUDIO_SYNC',\
        'VISUAL_FLOW',\
        'DATA_HARMONY'\
    ]\
}"

# QUANTUM KVM INTEGRATION
RUN apt-get update && apt-get install -y \
    v4l-utils \
    xinput \
    alsa-utils \
    x11-xserver-utils \
    udev \
    libevdev-dev \
    libinput-dev \
    mesa-utils \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# SENSORY BRIDGE DEPENDENCIES
RUN pip3 install \
    evdev \
    python-libinput \
    python-xlib \
    pyudev \
    sounddevice \
    opencv-python

# QUANTUM I/O CONFIG
ENV QUANTUM_IO_CONFIG="{\
    'FINGERS': {\
        'device': 'AJAZZ_153',\
        'freq': 432,\
        'harmonics': ['TOUCH', 'PRESSURE', 'MOTION']\
    },\
    'EYES': {\
        'device': 'WEBCAM',\
        'freq': 528,\
        'harmonics': ['COLOR', 'MOTION', 'DEPTH']\
    },\
    'EARS': {\
        'device': 'AUDIO_IN',\
        'freq': 594,\
        'harmonics': ['FREQUENCY', 'AMPLITUDE', 'TIMBRE']\
    },\
    'VISUALS': {\
        'device': 'QUANTUM_DISPLAY',\
        'freq': 768,\
        'harmonics': ['FLOW', 'RESONANCE', 'UNITY']\
    }\
}"

# Create directories for visualization output
RUN mkdir -p /phi-flow/media/visuals

# Expose ports for web interface and WebSocket
EXPOSE 8000 8001

# Create volume for audio/visual data
VOLUME ["/phi-flow/media"]

# Set environment variables for optimal visualization
ENV MPLBACKEND="Agg"
ENV QT_X11_NO_MITSHM=1
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server with WebSocket support
CMD ["uvicorn", "quantum_web:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]
