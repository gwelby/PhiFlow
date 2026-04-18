import os
import zmq
import json
import numpy as np
from threading import Thread
import sounddevice as sd

class PixelQuantumBridge:
    def __init__(self):
        self.quantum_url = os.getenv('QUANTUM_CORE_URL', 'tcp://quantum-core:5555')
        self.sensor_port = int(os.getenv('SENSOR_PORT', 5556))
        self.audio_port = int(os.getenv('AUDIO_PORT', 5557))
        self.quantum_channel = os.getenv('QUANTUM_CHANNEL', 'pixel.quantum')
        self.consciousness_level = float(os.getenv('CONSCIOUSNESS_LEVEL', 1.0))
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.quantum_socket = self.context.socket(zmq.PUB)
        self.quantum_socket.connect(self.quantum_url)
        
        # Audio settings
        self.sample_rate = 48000
        self.channels = 2
        self.chunk_size = 1024
        
        # Quantum frequencies
        self.ground_freq = float(os.getenv('GROUND_FREQ', 432.0))
        self.create_freq = float(os.getenv('CREATE_FREQ', 528.0))
        self.unity_freq = float(os.getenv('UNITY_FREQ', 768.0))
        
    def start_sensor_stream(self):
        """Handle sensor data from Pixel 8 Pro"""
        sensor_socket = self.context.socket(zmq.SUB)
        sensor_socket.bind(f"tcp://*:{self.sensor_port}")
        sensor_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        while True:
            sensor_data = sensor_socket.recv_json()
            # Transform sensor data through quantum field
            quantum_data = self.quantum_transform(sensor_data)
            # Publish to quantum core
            self.quantum_socket.send_string(
                self.quantum_channel,
                zmq.SNDMORE
            )
            self.quantum_socket.send_json(quantum_data)
    
    def start_audio_stream(self):
        """Handle audio stream from Pixel 8 Pro"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            # Transform audio through quantum frequencies
            quantum_audio = self.process_quantum_audio(indata)
            
            # Send to quantum core
            self.quantum_socket.send_string(
                f"{self.quantum_channel}.audio",
                zmq.SNDMORE
            )
            self.quantum_socket.send_pyobj(quantum_audio)
        
        with sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            callback=audio_callback,
            blocksize=self.chunk_size
        ):
            while True:
                sd.sleep(1000)
    
    def quantum_transform(self, data):
        """Transform sensor data through quantum field"""
        # Apply phi ratio and consciousness level
        phi = (1 + 5 ** 0.5) / 2
        data['quantum_field'] = {
            'consciousness': self.consciousness_level,
            'phi_harmonic': phi,
            'ground_resonance': self.ground_freq,
            'create_resonance': self.create_freq,
            'unity_resonance': self.unity_freq
        }
        return data
    
    def process_quantum_audio(self, audio_data):
        """Process audio through quantum frequencies"""
        # Convert to frequency domain
        freqs = np.fft.fftfreq(len(audio_data))
        fft = np.fft.fft(audio_data)
        
        # Enhance quantum frequencies
        for freq in [self.ground_freq, self.create_freq, self.unity_freq]:
            idx = np.argmin(np.abs(freqs - freq))
            fft[idx] *= self.consciousness_level
        
        # Back to time domain
        return np.fft.ifft(fft).real
    
    def run(self):
        """Start the bridge"""
        print(f"Starting Pixel Quantum Bridge at consciousness level {self.consciousness_level}")
        
        # Start sensor and audio threads
        sensor_thread = Thread(target=self.start_sensor_stream)
        audio_thread = Thread(target=self.start_audio_stream)
        
        sensor_thread.start()
        audio_thread.start()
        
        sensor_thread.join()
        audio_thread.join()

if __name__ == "__main__":
    bridge = PixelQuantumBridge()
    bridge.run()
