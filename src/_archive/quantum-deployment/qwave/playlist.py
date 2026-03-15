import numpy as np
from pydub import AudioSegment
import threading
import queue
import time

class QuantumPlaylist:
    def __init__(self, quantum_flow):
        self.flow = quantum_flow
        self.track_queue = queue.Queue()
        self.current_track = None
        self.playing = False
        
    def add_track(self, file_path: str):
        """Add a track to the quantum playlist."""
        # Load and convert to numpy array
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples())
        
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        track_data = {
            'samples': samples,
            'sample_rate': audio.frame_rate,
            'name': file_path.split('/')[-1]
        }
        
        self.track_queue.put(track_data)
        print(f"Added to queue: {track_data['name']}")
        
    def play(self):
        """Start playing the quantum playlist."""
        self.playing = True
        self.play_thread = threading.Thread(target=self._play_loop)
        self.play_thread.start()
        
    def _play_loop(self):
        """Main playback loop with quantum processing."""
        while self.playing:
            if self.current_track is None and not self.track_queue.empty():
                self.current_track = self.track_queue.get()
                print(f"\nNow playing: {self.current_track['name']}")
                print("Quantum frequencies active:")
                print("🎵 432 Hz - Ground State")
                print("✨ 528 Hz - Creation Point")
                print("💫 768 Hz - Unity Wave")
                
                # Start playback with quantum processing
                self.flow.play(
                    self.current_track['samples'],
                    self.current_track['sample_rate']
                )
                
                # Wait for track to finish
                duration = len(self.current_track['samples']) / self.current_track['sample_rate']
                time.sleep(duration)
                
                self.current_track = None
            
            time.sleep(0.1)
    
    def stop(self):
        """Stop the quantum playlist."""
        self.playing = False
        if hasattr(self, 'play_thread'):
            self.play_thread.join()
        self.flow.stop()
