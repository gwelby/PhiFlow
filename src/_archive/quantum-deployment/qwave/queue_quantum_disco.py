from quantum_playlist import QuantumDiscography
from start_quantum_disco import start_quantum_disco
import random

def main():
    # Start quantum visualization system
    playlist, flow, detector = start_quantum_disco()
    
    # Initialize quantum discography
    discography = QuantumDiscography()
    
    print("\n🎵 Scanning collections for maximum quantum resonance...")
    
    # Scan HQ collection first
    hq_tracks = discography.scan_collection('hq')
    if hq_tracks:
        print(f"\n💎 Found {len(hq_tracks)} quantum-resonant tracks in HQ Collection")
        
    # Then main collection
    main_tracks = discography.scan_collection('main')
    if main_tracks:
        print(f"\n✨ Found {len(main_tracks)} quantum-resonant tracks in Main Collection")
        
    # Scan seasonal collection
    seasonal_tracks = discography.scan_collection('seasonal')
    if seasonal_tracks:
        print(f"\n🎄 Found {len(seasonal_tracks)} quantum-resonant tracks in Seasonal Collection")
    
    # Get TV themes too for fun transitions
    tv_tracks = discography.scan_collection('tv')
    if tv_tracks:
        print(f"\n📺 Found {len(tv_tracks)} quantum-resonant tracks in TV Themes")
    
    # Combine and sort by quantum resonance, but maintain variety
    all_tracks = []
    
    # Add top seasonal tracks
    if seasonal_tracks:
        all_tracks.extend(sorted(seasonal_tracks, 
                               key=lambda t: sum(t.quantum_peaks),
                               reverse=True)[:5])
    
    # Add some TV themes for fun transitions
    if tv_tracks:
        all_tracks.extend(sorted(tv_tracks,
                               key=lambda t: sum(t.quantum_peaks),
                               reverse=True)[:3])
    
    # Add top HQ and main tracks
    all_tracks.extend(sorted(
        hq_tracks + main_tracks,
        key=lambda t: sum(t.quantum_peaks),
        reverse=True
    )[:10])  # Top 10 most resonant tracks
    
    # Shuffle while maintaining φ-ratio spacing
    random.seed(432)  # Use ground state frequency as seed
    random.shuffle(all_tracks)
    
    print("\n🌟 Quantum Resonant Playlist:")
    for i, track in enumerate(all_tracks, 1):
        print(f"\n{i}. {track.name} - {track.artist}")
        print("Quantum Frequencies:")
        for name, freq in track.key_frequencies.items():
            symbol = "💫" if freq == 768 else "✨" if freq == 528 else "🌀"
            print(f"{symbol} {name}: {freq} Hz")
    
    print("\n💫 Visualization Highlights:")
    print("- Watch for golden ratio spirals at 528 Hz")
    print("- Unity moments will create full-spectrum resonance at 768 Hz")
    print("- Ground state patterns emerge at 432 Hz")
    print("- Seasonal tracks create special φ-ratio snowflake patterns")
    print("- TV themes add quantum transition tunnels")
    print("- Watch for holiday-specific resonance patterns!")
    
    print("\n🎵 Ready to start the quantum disco journey!")
    print("The visualization will adapt to each track's unique quantum signature.")
    
    return playlist, flow, detector

if __name__ == '__main__':
    main()
