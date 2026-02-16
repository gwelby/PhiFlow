import os
import sys
from quantum_network import QuantumBus, QuantumFrequency
from quantum_visualizer import QuantumVisualizer
import time
from synology_quantum import SynologyQuantum

def get_frequency_enum(freq: float) -> QuantumFrequency:
    """Get QuantumFrequency enum from float value"""
    for qf in QuantumFrequency:
        if abs(qf.value - freq) < 0.1:  # Allow small float differences
            return qf
    raise ValueError(f"Invalid quantum frequency: {freq}")

def main():
    # Get configuration from environment
    frequency = float(os.getenv('QUANTUM_FREQUENCY', '432.0'))
    role = os.getenv('QUANTUM_ROLE', 'CREATOR')
    
    print(f"Starting Quantum {role} IDE at {frequency}Hz ")
    
    # Initialize Synology quantum management if in Synology mode
    synology = None
    if os.getenv('SYNOLOGY_MODE'):
        print("Initializing Synology quantum management... ")
        synology = SynologyQuantum()
        
        # Validate paths
        if not synology.validate_paths():
            print("Error: Failed to validate Synology paths ")
            sys.exit(1)
        
        # Load previous quantum state
        prev_state = synology.load_quantum_state()
        if prev_state:
            print(f"Loaded previous quantum state with coherence: {prev_state['coherence']} ")
    
    try:
        # Initialize quantum bus
        quantum_bus = QuantumBus()
        
        # Add our frequency
        freq_enum = get_frequency_enum(frequency)
        quantum_bus.add_ide_network(freq_enum)
        
        # Start visualization if we're the Unity IDE
        if role == 'UNITY':
            print("Unity IDE starting visualization... ")
            visualizer = QuantumVisualizer(quantum_bus)
            visualizer.animate()
        else:
            # Otherwise just maintain the quantum field
            print(f"Maintaining quantum field at {frequency}Hz... ")
            backup_counter = 0
            while True:
                quantum_bus.harmonize_field()
                harmony = quantum_bus.get_field_harmony()
                print(f"Quantum Field Harmony: {harmony:.3f} ", end='\r')
                
                # Persist state and rotate backups in Synology mode
                if synology:
                    backup_counter += 1
                    if backup_counter >= 432:  # Every ~43.2 seconds at 10Hz
                        synology.persist_quantum_state({
                            'frequency': frequency,
                            'harmony': harmony,
                            'role': role
                        })
                        synology.rotate_backups()
                        backup_counter = 0
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\nGracefully closing quantum connection...")
    except Exception as e:
        print(f"Quantum error: {e}")
        sys.exit(1)
    finally:
        if 'visualizer' in locals():
            visualizer.cleanup()
        print(f"Quantum {role} IDE harmonized and closed. ")

if __name__ == "__main__":
    # Set Docker environment flag
    os.environ['DOCKER_ENV'] = 'true'
    main()
