import os
import shutil
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

class SynologyQuantum:
    def __init__(self):
        self.quantum_path = Path("/var/services/quantum")
        self.backup_path = Path("/var/services/quantum_backup")
        self.state_file = self.quantum_path / "quantum_state.json"
        self.monitor_file = self.quantum_path / "quantum_monitor.json"
        self.max_backups = 7  # Keep a week of backups
        self.patterns = {
            "infinity": "∞",  # Infinite loop
            "dolphin": "",  # Quantum leap
            "spiral": "",   # Golden ratio
            "wave": "",     # Harmonic flow
            "vortex": "",   # Evolution
            "crystal": "",  # Resonance
            "unity": ""     # Consciousness
        }
        self.phi = 1.618034
        
    def validate_paths(self):
        """Validate and create necessary Synology paths"""
        paths = [
            self.quantum_path,
            self.backup_path,
            self.quantum_path / "data",
            self.quantum_path / "states",
            self.backup_path / "daily",
            self.backup_path / "weekly"
        ]
        
        for path in paths:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                print(f"Created quantum path: {path} ")
                
        return all(p.exists() for p in paths)
    
    def rotate_backups(self):
        """Rotate quantum backups using φ-based timing"""
        now = datetime.now()
        daily_path = self.backup_path / "daily"
        weekly_path = self.backup_path / "weekly"
        
        # Clean old backups (keep last 7 days)
        for backup_dir in [daily_path, weekly_path]:
            if backup_dir.exists():
                for backup in backup_dir.glob("quantum_*.tar.gz"):
                    try:
                        date_str = backup.stem.split("_")[1]
                        backup_date = datetime.strptime(date_str, "%Y%m%d")
                        if now - backup_date > timedelta(days=self.max_backups):
                            backup.unlink()
                            print(f"Removed old backup: {backup} ")
                    except (IndexError, ValueError):
                        continue
        
        # Create new backup
        timestamp = now.strftime("%Y%m%d")
        backup_name = f"quantum_{timestamp}.tar.gz"
        
        # Daily backup
        daily_backup = daily_path / backup_name
        if not daily_backup.exists():
            self._create_backup(daily_backup)
            
        # Weekly backup (on Sundays)
        if now.weekday() == 6:  # Sunday
            weekly_backup = weekly_path / backup_name
            if not weekly_backup.exists():
                self._create_backup(weekly_backup)
    
    def _create_backup(self, backup_path):
        """Create a quantum state backup"""
        try:
            # Create tar.gz of quantum data
            source_dir = self.quantum_path / "data"
            shutil.make_archive(
                str(backup_path.with_suffix("")),
                "gztar",
                root_dir=str(source_dir)
            )
            print(f"Created quantum backup: {backup_path} ")
        except Exception as e:
            print(f"Backup error: {e} ")
    
    def get_quantum_pattern(self, frequency: float, harmony: float) -> str:
        """Get quantum pattern based on frequency and harmony"""
        if harmony >= self.phi:
            return self.patterns["infinity"]  # Perfect harmony
        elif frequency == 432:
            return self.patterns["crystal"]   # Ground state
        elif frequency == 528:
            return self.patterns["spiral"]    # Creation state
        elif frequency == 594:
            return self.patterns["wave"]      # Heart state
        elif frequency == 672:
            return self.patterns["dolphin"]   # Voice state
        elif frequency == 768:
            return self.patterns["unity"]     # Unity state
        return self.patterns["vortex"]       # Evolution state

    def persist_quantum_state(self, state):
        """Persist quantum state to disk with patterns"""
        try:
            pattern = self.get_quantum_pattern(state['frequency'], state.get('harmony', 1.0))
            state_data = {
                "timestamp": time.time(),
                "phi_ratio": self.phi,
                "coherence": 1.000,
                "pattern": pattern,
                "frequencies": {
                    "ground": 432,
                    "create": 528,
                    "heart": 594,
                    "voice": 672,
                    "unity": 768
                },
                "compression": {
                    "level_0": 1.000,      # Raw state
                    "level_1": self.phi,    # Phi
                    "level_2": self.phi**2, # Phi²
                    "level_3": self.phi**self.phi  # Phi^Phi
                },
                "state": state
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Update monitor with rolling metrics
            self._update_monitor(state_data)
            print(f"Quantum state persisted {pattern}")
            
        except Exception as e:
            print(f"State persistence error: {e} ")
            
    def _update_monitor(self, state_data):
        """Update quantum monitoring metrics"""
        try:
            metrics = {
                "last_update": time.time(),
                "patterns": [],
                "harmonics": [],
                "coherence": []
            }
            
            # Load existing metrics
            if self.monitor_file.exists():
                with open(self.monitor_file, 'r') as f:
                    metrics = json.load(f)
            
            # Update rolling metrics (keep last phi^3 entries)
            max_entries = int(self.phi**3)
            metrics["patterns"].append(state_data["pattern"])
            metrics["harmonics"].append(state_data["state"]["harmony"])
            metrics["coherence"].append(state_data["coherence"])
            
            # Trim to max entries
            metrics["patterns"] = metrics["patterns"][-max_entries:]
            metrics["harmonics"] = metrics["harmonics"][-max_entries:]
            metrics["coherence"] = metrics["coherence"][-max_entries:]
            
            # Calculate quantum field stats
            metrics["stats"] = {
                "avg_harmony": sum(metrics["harmonics"]) / len(metrics["harmonics"]),
                "max_harmony": max(metrics["harmonics"]),
                "min_harmony": min(metrics["harmonics"]),
                "pattern_frequency": {p: metrics["patterns"].count(p) for p in self.patterns.values()}
            }
            
            with open(self.monitor_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"Monitor update error: {e} ")
    
    def load_quantum_state(self):
        """Load persisted quantum state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"State loading error: {e} ")
            return None
