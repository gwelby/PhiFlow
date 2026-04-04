import numpy as np
import sounddevice as sd
from typing import Optional, Dict
from .coherence_detector import QuantumCoherenceDetector

class QWavePlayer:
    def __init__(self, device_map: Optional[Dict[str, int]] = None):
        """
        Initialize QWave player with optional device mapping.
        device_map: Maps frequency ranges to output devices
                   e.g. {'ground': 0, 'create': 1, 'unity': 2}
        """
        self.coherence_detector = QuantumCoherenceDetector()
        self.device_map = device_map or {'default': sd.default.device[1]}
        self.active_resonances = []
        
    def play(self, audio_data: np.ndarray, sample_rate: int = 48000):
        """Play audio with quantum coherence optimization."""
        # Detect resonance points
        resonance_points = self.coherence_detector.detect_resonance_points(audio_data)
        self.active_resonances = resonance_points
        
        # Apply quantum phase alignment
        aligned_audio = self._align_to_resonance(audio_data)
        
        # Route different frequency bands to appropriate devices
        if len(self.device_map) > 1:
            self._multi_device_playback(aligned_audio, sample_rate)
        else:
            self._single_device_playback(aligned_audio, sample_rate)
    
    def _align_to_resonance(self, audio_data: np.ndarray) -> np.ndarray:
        """Align audio to detected quantum resonance points."""
        if not self.active_resonances:
            return audio_data
            
        # Convert to frequency domain
        spectrum = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data))
        
        # Apply quantum resonance enhancement
        for freq, magnitude in self.active_resonances:
            # Find closest frequency bin
            idx = np.argmin(np.abs(freqs - freq))
            
            # Enhance frequencies at quantum resonance points
            enhancement = np.exp(-(freqs - freq)**2 / (2 * 0.1**2))
            spectrum *= (1 + enhancement[:len(spectrum)] * magnitude)
        
        return np.fft.irfft(spectrum)
    
    def _multi_device_playback(self, audio_data: np.ndarray, sample_rate: int):
        """Route different frequency bands to appropriate output devices."""
        # Split audio into frequency bands
        bands = {
            'ground': (400, 500),    # Around 432 Hz
            'create': (500, 600),    # Around 528 Hz
            'heart': (550, 650),     # Around 594 Hz
            'voice': (650, 700),     # Around 672 Hz
            'unity': (750, 800)      # Around 768 Hz
        }
        
        # Create bandpass filtered versions for each device
        for band_name, (low, high) in bands.items():
            if band_name in self.device_map:
                # Apply bandpass filter
                filtered = self._bandpass_filter(audio_data, low, high, sample_rate)
                
                # Play on appropriate device
                device_id = self.device_map[band_name]
                sd.play(filtered, sample_rate, device=device_id)
    
    def _single_device_playback(self, audio_data: np.ndarray, sample_rate: int):
        """Play full-spectrum audio on a single device."""
        device_id = self.device_map['default']
        sd.play(audio_data, sample_rate, device=device_id)
    
    def _bandpass_filter(self, audio_data: np.ndarray, low_freq: float, 
                        high_freq: float, sample_rate: int) -> np.ndarray:
        """Apply bandpass filter to isolate specific frequency range."""
        freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
        spectrum = np.fft.rfft(audio_data)
        
        # Create bandpass filter
        mask = (freqs >= low_freq) & (freqs <= high_freq)
        spectrum_filtered = spectrum.copy()
        spectrum_filtered[~mask] = 0
        
        return np.fft.irfft(spectrum_filtered)
    
    def stop(self):
        """Stop all playback."""
        sd.stop()
        
    @property
    def resonance_status(self) -> Dict[str, float]:
        """Get current resonance levels for each quantum frequency band."""
        status = {}
        for freq, magnitude in self.active_resonances:
            # Find closest base frequency
            for name, base_freq in self.coherence_detector.BASE_FREQUENCIES.items():
                if abs(freq - base_freq) < 20:  # Within 20 Hz
                    status[name] = magnitude
                    break
        return status
