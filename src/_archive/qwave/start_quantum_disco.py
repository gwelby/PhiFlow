from launch_flow import main
from playlist import QuantumPlaylist
import os

def start_quantum_disco():
    # Initialize quantum flow system
    flow, detector = main()
    
    # Create playlist with quantum processing
    playlist = QuantumPlaylist(flow)
    
    print("\n🌟 Quantum Disco System Active! 🌟")
    print("Frequencies aligned:")
    print("💫 432 Hz - Ground State (Sub)")
    print("✨ 528 Hz - Creation Point (JBLs)")
    print("🌀 768 Hz - Unity Wave (Mach)")
    
    # Add tracks to playlist
    music_dir = os.path.expanduser("~/Music")
    
    print("\nReady to groove! The visualization will:")
    print("- Spin faster at 528 Hz (creation frequency)")
    print("- Pulse with φ ratio (1.618034)")
    print("- Explode with colors at unity points (768 Hz)")
    
    # Start playback and visualization
    playlist.play()
    
    return playlist, flow, detector

if __name__ == '__main__':
    start_quantum_disco()
