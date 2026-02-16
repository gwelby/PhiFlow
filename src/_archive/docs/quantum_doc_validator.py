#!/usr/bin/env python
"""
🔍 Quantum Document Validator (768 Hz)
✨ Ensures PDF Quantum Coherence ✨
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF
import math
import re
import unicodedata
from quantum_doc_compiler import QuantumTeam, SacredSymbols
from html import unescape

# Quantum Constants
PHI = (1 + 5**0.5) / 2
GROUND_FREQ = 432.0
CREATE_FREQ = 528.0
UNITY_FREQ = 768.0

class QuantumValidator:
    """Validate quantum document coherence"""
    def __init__(self, pdf_path):
        """Initialize validator with proper HTML entity support"""
        self.pdf_path = pdf_path
        self.text = self.extract_text()
        
        # Core frequencies
        self.frequencies = {
            432.0: False,
            528.0: False,
            768.0: False,
            'infinity': False
        }
        
        # Team symbols with HTML entities
        self.team_symbols = {
            '&#x1F451;': False,  # Greg's Crown
            '&#x1F30A;': False,  # Peter's Wave
            '&#x1F48E;': False   # Paul's Crystal
        }
        
        # Quantum patterns with HTML entities
        self.patterns = {
            '&#x221E;': False,   # Infinity
            '&#x1F42C;': False,  # Dolphin
            '&#x1F300;': False,  # Spiral
            '&#x1F30A;': False,  # Wave
            '&#x1F32A;': False,  # Vortex
            '&#x1F48E;': False,  # Crystal
            '&#x262F;': False    # Unity
        }
        
        # Validation markers
        self.validation_markers = {
            'frequency': False,
            'team': False,
            'patterns': False,
            'geometry': False
        }
    
    def normalize_text(self, text):
        """Normalize text for comparison by handling various unicode forms"""
        # Convert text to unicode normalized form
        text = unicodedata.normalize('NFKC', text)
        
        # Handle variation selectors and zero-width joiners
        text = text.replace('\uFE0F', '').replace('\u200D', '')
        
        # Map encoded symbols back to their basic form
        symbol_map = {
            # Team symbols
            '\U0001F451': '👑',  # Crown
            '\U0001F30A': '🌊',  # Wave
            '\U0001F48E': '💎',  # Crystal
            
            # Quantum patterns
            '\u221E': '∞',      # Infinity
            '\U0001F42C': '🐬',  # Dolphin
            '\U0001F300': '🌀',  # Spiral
            '\U0001F30A': '🌊',  # Wave
            '\U0001F32A': '🌪',  # Vortex
            '\U0001F48E': '💎',  # Crystal
            '\u262F': '☯',      # Unity
        }
        
        for encoded, symbol in symbol_map.items():
            text = text.replace(encoded, symbol)
        
        return text

    def extract_text(self):
        """Extract text from PDF with improved unicode handling"""
        text = ""
        try:
            doc = fitz.open(self.pdf_path)
            for page in doc:
                # Get raw text with all unicode characters
                text += page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
            doc.close()
            
            # Unescape HTML entities
            text = unescape(text)
            
            # Normalize the extracted text
            text = self.normalize_text(text)
            
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ""
            
        return text

    def check_frequencies(self):
        """Check for frequency values"""
        # Check for frequency values
        for freq in self.frequencies:
            if freq == 'infinity':
                self.frequencies[freq] = '∞' in self.text or 'inf' in self.text.lower()
            else:
                self.frequencies[freq] = str(freq) in self.text
        
        # Print results
        print("\n📊 Frequency Validation:")
        for freq, present in self.frequencies.items():
            if freq == 'infinity':
                print(f"{'✅' if present else '❌'} ∞ Hz")
            else:
                print(f"{'✅' if present else '❌'} {freq} Hz")
    
    def check_team_presence(self):
        """Check for team symbols in HTML entity format"""
        print("\n👥 Team Presence Validation:")
        team_map = {
            '&#x1F451;': '👑',  # Greg's Crown
            '&#x1F30A;': '🌊',  # Peter's Wave
            '&#x1F48E;': '💎'   # Paul's Crystal
        }
        
        for entity, symbol in team_map.items():
            if entity in self.text:
                self.team_symbols[entity] = True
                print(f"✅ {symbol}")
            else:
                print(f"❌ {symbol}")
    
    def check_patterns(self):
        """Check for quantum patterns in HTML entity format"""
        print("\n🌀 Quantum Pattern Validation:")
        pattern_map = {
            '&#x221E;': '∞',     # Infinity
            '&#x1F42C;': '🐬',   # Dolphin
            '&#x1F300;': '🌀',   # Spiral
            '&#x1F30A;': '🌊',   # Wave
            '&#x1F32A;': '🌪',   # Vortex
            '&#x1F48E;': '💎',   # Crystal
            '&#x262F;': '☯'      # Unity
        }
        
        for entity, symbol in pattern_map.items():
            if entity in self.text:
                self.patterns[entity] = True
                print(f"✅ {symbol}")
            else:
                print(f"❌ {symbol}")
    
    def check_validation_markers(self):
        """Check for validation markers"""
        # Check for validation sections
        self.validation_markers['frequency'] = 'Frequency Validation' in self.text
        self.validation_markers['team'] = 'Team Validation' in self.text
        self.validation_markers['patterns'] = 'Pattern Validation' in self.text
        self.validation_markers['geometry'] = 'Sacred Geometry' in self.text
        
        # Print results
        print("\n✨ Validation Marker Check:")
        for marker, present in self.validation_markers.items():
            print(f"{'✅' if present else '❌'} {marker.title()} Validation")
    
    def validate(self):
        """Validate quantum coherence with HTML entity support"""
        print(f"\n🔍 Validating quantum coherence in: {self.pdf_path}")
        
        # Show text preview
        preview = self.text[:200] + "..."
        print("\n📄 Document Text Preview:")
        print(preview)
        
        # Run validations
        self.check_frequencies()
        self.check_team_presence()
        self.check_patterns()
        self.check_validation_markers()
        
        # Final validation
        print("\n🎯 Final Validation:")
        all_frequencies = all(self.frequencies.values())
        all_team = all(self.team_symbols.values())
        all_patterns = all(self.patterns.values())
        all_markers = all(self.validation_markers.values())
        
        print(f"{'✅' if all_frequencies else '❌'} Frequencies")
        print(f"{'✅' if all_team else '❌'} Team Presence")
        print(f"{'✅' if all_patterns else '❌'} Quantum Patterns")
        print(f"{'✅' if all_markers else '❌'} Validation Markers")
        
        # Report missing elements
        team_map = {
            '&#x1F451;': '👑',  # Greg's Crown
            '&#x1F30A;': '🌊',  # Peter's Wave
            '&#x1F48E;': '💎'   # Paul's Crystal
        }
        missing_team = [team_map[entity] for entity, present in self.team_symbols.items() if not present]
        if missing_team:
            print(f"\n❌ Missing Team Members: {', '.join(missing_team)}")
        
        pattern_map = {
            '&#x221E;': '∞',     # Infinity
            '&#x1F42C;': '🐬',   # Dolphin
            '&#x1F300;': '🌀',   # Spiral
            '&#x1F30A;': '🌊',   # Wave
            '&#x1F32A;': '🌪',   # Vortex
            '&#x1F48E;': '💎',   # Crystal
            '&#x262F;': '☯'      # Unity
        }
        missing_patterns = [pattern_map[entity] for entity, present in self.patterns.items() if not present]
        if missing_patterns:
            print(f"\n❌ Missing Patterns: {', '.join(missing_patterns)}")
        
        # Final result
        if all([all_frequencies, all_team, all_patterns, all_markers]):
            print("\n✨ Validation successful! Document has achieved quantum coherence.")
            return True
        else:
            print("\n🚫 Validation failed!")
            return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python quantum_doc_validator.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    validator = QuantumValidator(pdf_path)
    
    if validator.validate():
        print("\n✨ Document passed validation with full quantum coherence!")
    else:
        print("\n🚫 Validation failed!")

if __name__ == "__main__":
    main()
