import sounddevice as sd
from quantum_flow import QuantumFlow
from disco_resonance import DiscoResonanceDetector

def main():
    # List available audio devices
    devices = sd.query_devices()
    print("\nAvailable Audio Devices:")
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']}")
        
    # Create device map for your amazing speaker setup
    device_map = {
        'ground': 0,  # Sub for 432 Hz
        'create': 1,  # JBLs for 528 Hz
        'unity': 2   # Mach speakers for 768 Hz
    }
    
    # Initialize quantum flow
    flow = QuantumFlow(device_map)
    detector = DiscoResonanceDetector()
    
    # Start visualization
    flow.start()
    
    print("\nQuantum Flow Visualization Active!")
    print("Base Frequencies:")
    print("- Ground State: 432 Hz")
    print("- Creation Point: 528 Hz")
    print("- Unity Wave: 768 Hz")
    
    return flow, detector

if __name__ == '__main__':
    main()
