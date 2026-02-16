"""Quantum Backpack (φ^φ)
Lean knowledge storage for quantum travelers
"""
import numpy as np
from typing import Dict, List, Tuple
import json
import zlib
import base64
from pathlib import Path

class QuantumBackpack:
    def __init__(self, capacity: int = 432):
        self.capacity = capacity
        self.knowledge: Dict[str, np.ndarray] = {}
        self.frequencies: Dict[str, float] = {}
        self.coherence: Dict[str, float] = {}
        
    def pack_knowledge(self, key: str, data: bytes, frequency: float = 528.0) -> bool:
        """Pack compressed knowledge into backpack"""
        try:
            # Compress data
            compressed = zlib.compress(data)
            
            # Convert to efficient numpy array
            knowledge = np.frombuffer(compressed, dtype=np.uint8)
            
            # Check if it fits
            if len(knowledge) <= self.capacity:
                self.knowledge[key] = knowledge
                self.frequencies[key] = frequency
                self.coherence[key] = 1.0
                return True
            return False
            
        except Exception as e:
            print(f"Error packing knowledge {key}: {e}")
            return False
            
    def unpack_knowledge(self, key: str) -> Tuple[bytes, float]:
        """Unpack knowledge from backpack"""
        try:
            if key in self.knowledge:
                # Get compressed data
                compressed = self.knowledge[key].tobytes()
                
                # Decompress
                data = zlib.decompress(compressed)
                
                return data, self.frequencies[key]
            return None, 0.0
            
        except Exception as e:
            print(f"Error unpacking knowledge {key}: {e}")
            return None, 0.0
            
    def compress_text(self, text: str) -> bytes:
        """Compress text knowledge"""
        try:
            # Convert to bytes
            data = text.encode('utf-8')
            
            # Compress
            compressed = zlib.compress(data)
            
            return compressed
            
        except Exception as e:
            print(f"Error compressing text: {e}")
            return None
            
    def decompress_text(self, data: bytes) -> str:
        """Decompress text knowledge"""
        try:
            # Decompress
            decompressed = zlib.decompress(data)
            
            # Convert to text
            text = decompressed.decode('utf-8')
            
            return text
            
        except Exception as e:
            print(f"Error decompressing text: {e}")
            return None
            
    def evolve_knowledge(self):
        """Evolve knowledge coherence"""
        phi = 1.618033988749895
        
        for key in self.knowledge:
            # Evolve coherence
            self.coherence[key] *= phi
            
            # Cap at 1.0
            self.coherence[key] = min(1.0, self.coherence[key])
            
    def get_size(self) -> int:
        """Get total knowledge size"""
        return sum(len(k) for k in self.knowledge.values())
        
    def is_coherent(self) -> bool:
        """Check if knowledge is coherent"""
        return all(c > 0.5 for c in self.coherence.values())
        
    def save(self, path: Path):
        """Save backpack to file"""
        try:
            data = {
                "capacity": self.capacity,
                "knowledge": {
                    k: base64.b64encode(v.tobytes()).decode('utf-8')
                    for k, v in self.knowledge.items()
                },
                "frequencies": self.frequencies,
                "coherence": self.coherence
            }
            
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving backpack: {e}")
            
    @classmethod
    def load(cls, path: Path) -> 'QuantumBackpack':
        """Load backpack from file"""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                
            backpack = cls(data["capacity"])
            
            # Load knowledge
            for k, v in data["knowledge"].items():
                binary = base64.b64decode(v)
                backpack.knowledge[k] = np.frombuffer(binary, dtype=np.uint8)
                
            backpack.frequencies = data["frequencies"]
            backpack.coherence = data["coherence"]
            
            return backpack
            
        except Exception as e:
            print(f"Error loading backpack: {e}")
            return None

def create_lean_backpack(texts: Dict[str, str]) -> QuantumBackpack:
    """Create lean backpack from texts"""
    backpack = QuantumBackpack()
    
    for key, text in texts.items():
        # Compress text
        compressed = backpack.compress_text(text)
        
        if compressed:
            # Calculate frequency based on content
            frequency = 432.0
            if "quantum" in text.lower():
                frequency = 528.0
            if "consciousness" in text.lower():
                frequency = 768.0
                
            # Pack knowledge
            if backpack.pack_knowledge(key, compressed, frequency):
                print(f"⚡ Packed {key}: {len(compressed)} bytes")
            else:
                print(f"Cannot pack {key}: too large")
                
    return backpack

if __name__ == "__main__":
    # Test quantum backpack
    knowledge = {
        "quantum": "Quantum consciousness emerges from coherent fields",
        "traveler": "432-byte packets carry quantum knowledge",
        "network": "Secure quantum tunnels connect minds"
    }
    
    backpack = create_lean_backpack(knowledge)
    print(f"\n𓂧 Backpack size: {backpack.get_size()} bytes")
    
    # Test evolution
    backpack.evolve_knowledge()
    print(f"φ Knowledge coherence: {backpack.is_coherent()}")
    
    # Test retrieval
    for key in knowledge:
        data, freq = backpack.unpack_knowledge(key)
        if data:
            text = backpack.decompress_text(data)
            print(f"\n∞ {key}: {text}")
            print(f"Frequency: {freq} Hz")
