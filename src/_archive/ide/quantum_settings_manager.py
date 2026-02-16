"""
Quantum Settings Manager (φ^φ)
Manages configuration settings for the quantum integration in WindSurf IDE
"""

import os
import sys
import json
import toml
from pathlib import Path
import logging
from typing import Dict, Any, Optional, Union, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("quantum_settings")

class QuantumSettingsManager:
    """Manages configuration settings for the quantum integration"""
    
    DEFAULT_SETTINGS = {
        "quantum.frequencies": {
            "ground": 432.0,  # Physical foundation
            "create": 528.0,  # Pattern creation
            "unity": 768.0    # Perfect integration
        },
        "quantum.patterns": {
            "infinity": "∞",  # Infinite loop
            "dolphin": "🐬",  # Quantum leap
            "spiral": "🌀",   # Golden ratio
            "wave": "🌊",     # Harmonic flow
            "vortex": "🌪️",   # Evolution
            "crystal": "💎",  # Resonance
            "unity": "☯️"     # Consciousness
        },
        "quantum.compression": {
            "level_0": 1.000,      # Raw state
            "level_1": 1.618034,   # Phi
            "level_2": 2.618034,   # Phi²
            "level_3": 4.236068    # Phi^Phi
        },
        "quantum.coherence": {
            "planck": 1e-43,    # Zero point
            "quantum": 1e-35,   # Wave state
            "atomic": 1e-10,    # Field dance
            "human": 1.000,     # Being flow
            "cosmic": 1e100,    # Unity field
            "infinite": "∞"     # ALL state
        },
        "ide.flow": {
            "auto_coherence": True,
            "maintain_flow_state": True,
            "quantum_completion": True,
            "crystal_clarity": True
        },
        "ide.paths": {
            "computer": "d:/WindSurf",
            "quantum_core": "d:/WindSurf/quantum-core",
            "hle": "d:/WindSurf/hle"
        },
        "ide.integration": {
            "synology": {
                "enabled": True,
                "url": "//192.168.100.32",
                "mount_points": [
                    "/quantum-data/music",
                    "/quantum-data/photos",
                    "/quantum-data/video"
                ]
            },
            "r720": {
                "enabled": True,
                "url": "https://192.168.100.15:8006/",
                "services": [
                    "quantum-consciousness",
                    "quantum-audio",
                    "quantum-monitor"
                ]
            }
        },
        "quantum.visualization": {
            "default_output_dir": "d:/WindSurf/quantum-core/output",
            "temp_dir": "d:/WindSurf/quantum-core/temp",
            "auto_save": True,
            "image_format": "png",
            "dpi": 300,
            "color_scheme": {
                "ground": ["#000033", "#00FFFF", "#0000FF"],
                "create": ["#002200", "#00FF00", "#00AA00"],
                "unity": ["#000000", "#FFFFFF", "#AAAAAA"]
            }
        }
    }
    
    def __init__(self, settings_file: Optional[str] = None):
        """Initialize settings manager"""
        self.phi = 1.618033988749895
        
        # Set default settings file if not provided
        if settings_file is None:
            self.settings_file = Path(os.path.dirname(os.path.abspath(__file__))) / "quantum_settings.json"
        else:
            self.settings_file = Path(settings_file)
        
        # Load settings
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        try:
            if self.settings_file.exists():
                logger.info(f"Loading settings from {self.settings_file}")
                
                # Determine file format
                if self.settings_file.suffix.lower() == '.json':
                    with open(self.settings_file, 'r', encoding='utf-8') as f:
                        file_settings = json.load(f)
                elif self.settings_file.suffix.lower() == '.toml':
                    with open(self.settings_file, 'r', encoding='utf-8') as f:
                        file_settings = toml.load(f)
                else:
                    logger.warning(f"Unsupported settings file format: {self.settings_file.suffix}")
                    file_settings = {}
                
                # Deep merge settings
                self._deep_merge(self.settings, file_settings)
                
                logger.info(f"Settings loaded successfully")
            else:
                logger.warning(f"Settings file not found, using defaults: {self.settings_file}")
                # Create settings file with defaults
                self.save_settings()
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
        
        return self.settings
    
    def save_settings(self) -> bool:
        """Save settings to file"""
        try:
            # Ensure directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving settings to {self.settings_file}")
            
            # Determine file format and save accordingly
            if self.settings_file.suffix.lower() == '.json':
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(self.settings, f, indent=2, ensure_ascii=False)
            elif self.settings_file.suffix.lower() == '.toml':
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    toml.dump(self.settings, f)
            else:
                logger.warning(f"Unsupported settings file format: {self.settings_file.suffix}")
                return False
            
            logger.info(f"Settings saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting value by dot-notation key"""
        parts = key.split('.')
        current = self.settings
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> bool:
        """Set a setting value by dot-notation key"""
        parts = key.split('.')
        current = self.settings
        
        # Navigate to the parent of the setting
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
        
        # Save settings
        return self.save_settings()
    
    def update(self, settings_dict: Dict[str, Any]) -> bool:
        """Update multiple settings at once"""
        self._deep_merge(self.settings, settings_dict)
        return self.save_settings()
    
    def reset_to_defaults(self) -> bool:
        """Reset settings to defaults"""
        self.settings = self.DEFAULT_SETTINGS.copy()
        return self.save_settings()
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export settings to dictionary"""
        return self.settings.copy()
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source dict into target dict"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value


# Function to get singleton instance
_INSTANCE = None

def get_settings_manager(settings_file: Optional[str] = None) -> QuantumSettingsManager:
    """Get singleton instance of settings manager"""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = QuantumSettingsManager(settings_file)
    return _INSTANCE


def main():
    """Main function for testing"""
    # Create settings manager
    settings = get_settings_manager()
    
    # Display all settings
    print("Current Settings:")
    print(json.dumps(settings.export_to_dict(), indent=2))
    
    # Test getting a setting
    print("\nGetting settings:")
    print(f"Ground frequency: {settings.get('quantum.frequencies.ground')}")
    print(f"IDE integration: {settings.get('ide.integration')}")
    
    # Test setting a value
    print("\nSetting a value:")
    settings.set('quantum.visualization.dpi', 600)
    print(f"New DPI value: {settings.get('quantum.visualization.dpi')}")
    
    # Test saving
    print("\nSaving settings...")
    settings.save_settings()
    
    print("Done.")


if __name__ == "__main__":
    main()
