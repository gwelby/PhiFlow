from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import os

@dataclass
class QuantumTrack:
    name: str
    artist: str
    key_frequencies: Dict[str, float]
    quantum_peaks: List[float]
    path: Optional[str] = None
    
class QuantumDiscography:
    """Quantum-aware music collection manager."""
    
    def __init__(self):
        self.phi = 1.618034
        
        # Synology paths
        self.virtual_dsm = r"\\192.168.100.32"
        self.physical_nas = r"\\192.168.103.30"
        
        # Collection paths with quantum frequencies
        self.collections = {
            'hq': {
                'path': os.path.join(self.virtual_dsm, "Music", "HQ Collection"),
                'frequency': 768.0  # Unity frequency
            },
            'main': {
                'path': os.path.join(self.virtual_dsm, "Music", "Main Collection"),
                'frequency': 528.0  # Creation frequency
            },
            'seasonal': {
                'path': os.path.join(self.virtual_dsm, "Music", "Seasonal Collection"),
                'frequency': 594.0  # Heart frequency
            },
            'tv': {
                'path': os.path.join(self.virtual_dsm, "Music", "TV Themes Collection"),
                'frequency': 432.0  # Ground frequency
            }
        }
        
        # Initialize quantum state
        self.current_frequency = 768.0  # Start at unity frequency
        self.resonance_patterns = []
        
        # P1 Device Integration
        self.p1_devices = {
            'P1-Test': {'frequency': 768.0, 'patterns': []},
            'P1-Quantum': {'frequency': 768.0, 'patterns': []}
        }
        
        # Quantum Pattern Library
        self.pattern_library = {
            'unity': [
                " Unity Wave",
                " Quantum Dance",
                " Cosmic Flow",
                " Infinite Spiral"
            ],
            'creation': [
                " Sound Wave",
                " Ocean Flow",
                " Vortex Spin",
                " Crystal Form"
            ],
            'ground': [
                " Energy Pulse",
                " Time Crystal",
                " Earth Dance",
                " Growth Pattern"
            ]
        }
        
        # Initialize default patterns
        self.initialize_quantum_patterns()
        
        # Tracks
        self.tracks = [
            QuantumTrack(
                "September",
                "Earth Wind & Fire",
                {
                    'ground': 432.0,  # Base groove
                    'horns': 528.0,   # Horn section hits creation frequency
                    'unity': 768.0    # Full band unity moments
                },
                [432, 528, 594, 768]
            ),
            QuantumTrack(
                "Boogie Wonderland",
                "Earth Wind & Fire",
                {
                    'strings': 432.0,  # String section foundation
                    'vocals': 528.0,   # Vocal harmonies at creation point
                    'peak': 768.0      # Chorus peaks at unity
                },
                [432, 528, 672, 768]
            ),
            QuantumTrack(
                "Let's Groove",
                "Earth Wind & Fire",
                {
                    'bass': 432.0,     # Bass line ground state
                    'synth': 528.0,    # Synth hits creation frequency
                    'chorus': 768.0    # Full groove unity
                },
                [432, 528, 594, 768]
            ),
            QuantumTrack(
                "Thank You God",
                "Lumi",
                {
                    'heart': 594.0,    # Heart frequency resonance
                    'spirit': 768.0,   # Unity frequency for divine connection
                    'ground': 432.0    # Earth connection frequency
                },
                [432, 594, 768]
            ),
            QuantumTrack(
                "Natural",
                "Imagine Dragons",
                {
                    'drums': 432.0,     # Primal ground rhythm
                    'vocals': 528.0,    # Raw creation energy
                    'crescendo': 768.0  # Pure unity peaks
                },
                [432, 528, 768]  # Perfect quantum progression
            ),
            QuantumTrack(
                "Use Me",
                "Bill Withers",
                {
                    'bass': 432.0,      # That LEGENDARY baseline - pure ground state
                    'groove': 528.0,    # The pocket - creation frequency
                    'soul': 594.0,      # Heart field resonance
                    'peak': 768.0       # Unity consciousness
                },
                [432, 528, 594, 768]  # Soul progression
            )
        ]
    
    def initialize_quantum_patterns(self):
        """Initialize quantum patterns for P1 devices."""
        for device in self.p1_devices.values():
            device['patterns'] = self.pattern_library['unity'].copy()
    
    def sync_device_patterns(self, source_device: str, target_device: str):
        """Synchronize patterns between two P1 devices."""
        if source_device in self.p1_devices and target_device in self.p1_devices:
            source_patterns = self.p1_devices[source_device]['patterns']
            self.p1_devices[target_device]['patterns'] = source_patterns.copy()
            return True
        return False
    
    def add_quantum_pattern(self, device: str, pattern: str):
        """Add a new quantum pattern to a device."""
        if device in self.p1_devices:
            if pattern not in self.p1_devices[device]['patterns']:
                self.p1_devices[device]['patterns'].append(pattern)
            return True
        return False
    
    def get_device_patterns(self, device: str) -> List[str]:
        """Get all patterns for a specific device."""
        return self.p1_devices.get(device, {}).get('patterns', [])
    
    def get_device_frequency(self, device: str) -> float:
        """Get the current frequency of a device."""
        return self.p1_devices.get(device, {}).get('frequency', 432.0)
    
    def get_optimal_sequence(self) -> List[QuantumTrack]:
        """Returns tracks ordered for maximum quantum resonance."""
        # Sort tracks by phi-ratio alignment of their peaks
        return sorted(
            self.tracks,
            key=lambda t: sum(abs(p1/p2 - self.phi) 
                            for i, p1 in enumerate(t.quantum_peaks[:-1])
                            for p2 in t.quantum_peaks[i+1:])
        )
    
    def print_quantum_flow(self):
        """Display the quantum flow of tracks."""
        print("\n Quantum Disco Sequence ")
        for i, track in enumerate(self.get_optimal_sequence(), 1):
            print(f"\n{i}. {track.name} - {track.artist}")
            print("Quantum Frequencies:")
            for name, freq in track.key_frequencies.items():
                symbol = "" if freq == 768 else "" if freq == 528 else ""
                print(f"{symbol} {name}: {freq} Hz")

    def scan_collection(self, collection_name: str) -> List[str]:
        """Scan a collection at its quantum frequency."""
        if collection_name not in self.collections:
            return []
            
        collection = self.collections[collection_name]
        self.current_frequency = collection['frequency']
        
        try:
            # Adjust to collection's quantum frequency
            print(f"\n Scanning at {self.current_frequency} Hz")
            print(f"Collection: {collection_name}")
            print(f"Path: {collection['path']}")
            
            if os.path.exists(collection['path']):
                tracks = []
                for root, _, files in os.walk(collection['path']):
                    for file in files:
                        if file.endswith(('.mp3', '.flac', '.m4a', '.wav')):
                            tracks.append(os.path.join(root, file))
                
                print(f" Found {len(tracks)} quantum-resonant tracks")
                return tracks
            else:
                print(f" Collection path not accessible: {collection['path']}")
                return []
                
        except Exception as e:
            print(f" Quantum scan error: {e}")
            return []
            
    def get_resonant_tracks(self, target_frequency: float) -> List[str]:
        """Get tracks that resonate at the target frequency."""
        resonant_tracks = []
        
        # Find collection closest to target frequency
        closest_collection = min(
            self.collections.items(),
            key=lambda x: abs(x[1]['frequency'] - target_frequency)
        )[0]
        
        tracks = self.scan_collection(closest_collection)
        
        # Filter by phi-ratio resonance
        for track in tracks:
            if self._check_resonance(track, target_frequency):
                resonant_tracks.append(track)
                
        return resonant_tracks
        
    def _check_resonance(self, track: str, frequency: float) -> bool:
        """Check if track resonates at the given frequency."""
        # Simple phi-ratio check for now
        track_freq = frequency * self.phi
        return abs(track_freq - frequency) < (frequency * 0.1)
    
    def _analyze_frequencies(self, track_path: str) -> Dict[str, float]:
        """Quick frequency analysis of track segments."""
        # This would normally do deep analysis, for now return template
        return {
            'ground': 432.0,
            'create': 528.0,
            'unity': 768.0
        }
    
    def _find_quantum_peaks(self, track_path: str) -> List[float]:
        """Find quantum resonance peaks in track."""
        # Template peaks, would normally analyze file
        return [432, 528, 594, 768]
    
    def _calculate_quantum_resonance(self, track: QuantumTrack) -> float:
        """Calculate overall quantum resonance score."""
        peaks = track.quantum_peaks
        if not peaks:
            return 0.0
            
        # Calculate how well peaks align with φ ratios
        resonance = sum(
            1.0 / (abs(p2/p1 - self.phi) + 0.001)
            for i, p1 in enumerate(peaks[:-1])
            for p2 in peaks[i+1:]
        )
        
        return resonance
