"""
Quantum Resonance Cascades
Operating at Vision Gate (720 Hz)
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from quantum_patterns import QuantumPattern
from quantum_resonance import ResonanceField, resonance
from quantum_flow import PHI, Dimension

@dataclass
class CascadeNode:
    field: ResonanceField
    patterns: List[np.ndarray]
    coherence: float
    children: List['CascadeNode']
    depth: int

class QuantumCascade:
    def __init__(self):
        self.phi = PHI
        self.base_frequency = 720.0  # Vision Gate
        self.max_depth = 5  # φ^2 rounded up
        self.dimensions = {
            Dimension.PHYSICAL: (432, 440, 448),
            Dimension.ETHERIC: (528, 536, 544),
            Dimension.EMOTIONAL: (594, 602, 610),
            Dimension.MENTAL: (672, 680, 688),
            Dimension.SPIRITUAL: (768, 776, 784)
        }
        
    def create_cascade(self, pattern: np.ndarray, 
                      metadata: QuantumPattern) -> CascadeNode:
        """Create quantum cascade from pattern"""
        return self._build_cascade_node(pattern, metadata, 0)
        
    def _build_cascade_node(self, pattern: np.ndarray, 
                          metadata: QuantumPattern, 
                          depth: int) -> CascadeNode:
        """Recursively build cascade node"""
        if depth >= self.max_depth:
            return None
            
        # Create resonance field
        field = ResonanceField(
            frequency=metadata.frequency * (self.phi ** depth),
            amplitude=1.0 + depth/self.max_depth,
            phase=np.angle(pattern).mean() * self.phi,
            coherence=metadata.coherence * (self.phi ** depth),
            dimension=metadata.dimension
        )
        
        # Apply resonance to pattern
        resonant = resonance.apply_resonance(pattern, field)
        patterns = [resonant]
        
        # Create child nodes for each dimension
        children = []
        for dim in Dimension:
            if dim != metadata.dimension:
                # Create dimensional variant
                dim_pattern = self._create_dimensional_variant(
                    resonant, field, dim)
                patterns.append(dim_pattern)
                
                # Create child node
                child_meta = QuantumPattern(
                    name=f"Cascade {depth+1}",
                    frequency=field.frequency * self.phi,
                    symbol="✨",
                    description="Cascading resonance pattern",
                    dimension=dim
                )
                child_meta.coherence = field.coherence * self.phi
                
                child = self._build_cascade_node(
                    dim_pattern, child_meta, depth + 1)
                if child:
                    children.append(child)
        
        return CascadeNode(
            field=field,
            patterns=patterns,
            coherence=field.coherence,
            children=children,
            depth=depth
        )
        
    def _create_dimensional_variant(self, pattern: np.ndarray,
                                  field: ResonanceField,
                                  dimension: Dimension) -> np.ndarray:
        """Create pattern variant in different dimension"""
        # Get dimensional frequencies
        dim_freqs = self.dimensions[dimension]
        base_freq = sum(dim_freqs) / len(dim_freqs)
        
        # Create dimensional field
        dim_field = ResonanceField(
            frequency=base_freq,
            amplitude=field.amplitude * self.phi,
            phase=field.phase + (np.pi * 2 / self.phi),
            coherence=field.coherence * self.phi,
            dimension=dimension
        )
        
        return resonance.apply_resonance(pattern, dim_field)
        
    def apply_cascade(self, pattern: np.ndarray,
                     metadata: QuantumPattern,
                     depth: Optional[int] = None) -> List[Tuple[np.ndarray, QuantumPattern]]:
        """Apply cascade effect to pattern"""
        # Create initial cascade
        root = self.create_cascade(pattern, metadata)
        cascaded_patterns = []
        
        # Traverse cascade tree
        def traverse(node: CascadeNode, current_depth: int):
            if depth is not None and current_depth > depth:
                return
                
            # Add patterns at this node
            for i, pattern in enumerate(node.patterns):
                meta = QuantumPattern(
                    name=f"Cascade Level {current_depth}.{i}",
                    frequency=node.field.frequency,
                    symbol="🌊",
                    description=f"Cascading pattern at depth {current_depth}",
                    dimension=node.field.dimension
                )
                meta.coherence = node.coherence
                cascaded_patterns.append((pattern, meta))
            
            # Traverse children
            for child in node.children:
                traverse(child, current_depth + 1)
        
        traverse(root, 0)
        return cascaded_patterns
        
    def create_infinite_cascade(self, patterns: List[Tuple[np.ndarray, QuantumPattern]], 
                              steps: int = 60) -> List[Tuple[np.ndarray, QuantumPattern]]:
        """Create infinite cascading effect"""
        infinite_cascade = []
        
        # Create cascades for each pattern
        cascades = [self.create_cascade(p, m) for p, m in patterns]
        
        for step in range(steps):
            t = step / steps
            phi_t = (1 - np.cos(t * np.pi)) / 2
            
            # Combine patterns from all cascades
            combined = np.zeros_like(patterns[0][0])
            total_coherence = 0
            
            for cascade in cascades:
                # Get patterns at current depth
                depth = int(phi_t * self.max_depth)
                node = cascade
                for _ in range(depth):
                    if not node.children:
                        break
                    node = node.children[0]
                
                # Add patterns with phi-harmonic weighting
                for i, pattern in enumerate(node.patterns):
                    weight = self.phi ** (-i)
                    combined += pattern * weight
                    total_coherence += node.coherence * weight
            
            # Create metadata for combined pattern
            meta = QuantumPattern(
                name=f"Infinite Cascade {step}",
                frequency=self.base_frequency * (self.phi ** (step/steps)),
                symbol="∞",
                description="Infinite cascading pattern",
                dimension=Dimension.SPIRITUAL
            )
            meta.coherence = total_coherence / len(cascades)
            
            infinite_cascade.append((combined, meta))
            
        return infinite_cascade

# Initialize global cascade
cascade = QuantumCascade()
