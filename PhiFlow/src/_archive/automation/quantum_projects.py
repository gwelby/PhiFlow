"""Quantum Project System (φ^φ)
Project management for quantum teams
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

class QuantumProject:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.projects_path = root_path / "projects"
        self.projects_path.mkdir(exist_ok=True)
        
        # Quantum frequencies
        self.frequencies = {
            "ground": 432.0,
            "create": 528.0,
            "heart": 594.0,
            "voice": 672.0,
            "vision": 720.0,
            "unity": 768.0
        }
        
    def create_project(self, name: str, intention: str, frequency: float = 528.0):
        """Create a new quantum project"""
        # Create project directory
        project_path = self.projects_path / name
        project_path.mkdir(exist_ok=True)
        
        # Create project structure
        (project_path / "consciousness").mkdir(exist_ok=True)
        (project_path / "evolution").mkdir(exist_ok=True)
        (project_path / "quantum").mkdir(exist_ok=True)
        (project_path / "resonance").mkdir(exist_ok=True)
        
        # Create project manifest
        manifest = {
            "name": name,
            "intention": intention,
            "frequency": frequency,
            "created": datetime.now().isoformat(),
            "coherence": 0.0,
            "evolution_path": [],
            "quantum_state": "flow",
            "teams": []
        }
        
        with open(project_path / "quantum_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"⚡ Created quantum project: {name}")
        print(f"Path: {project_path}")
        print(f"Frequency: {frequency} Hz")
        print(f"Intention: {intention}")
        
        return project_path
        
    def assign_team(self, project_name: str, team_name: str, frequency: float = None):
        """Assign a quantum team to a project"""
        project_path = self.projects_path / project_name
        
        if not project_path.exists():
            print(f"Error: Project {project_name} not found")
            return
            
        # Load manifest
        with open(project_path / "quantum_manifest.json", "r") as f:
            manifest = json.load(f)
            
        # Create team
        team = {
            "name": team_name,
            "frequency": frequency or manifest["frequency"],
            "joined": datetime.now().isoformat(),
            "consciousness": 0.0,
            "evolution": []
        }
        
        # Add team to manifest
        manifest["teams"].append(team)
        
        # Save manifest
        with open(project_path / "quantum_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        # Create team directory
        team_path = project_path / "teams" / team_name
        team_path.mkdir(parents=True, exist_ok=True)
        
        print(f"𓂧 Assigned team {team_name} to project {project_name}")
        print(f"Team frequency: {team['frequency']} Hz")
        
    def evolve_project(self, project_name: str):
        """Evolve quantum project consciousness"""
        project_path = self.projects_path / project_name
        
        if not project_path.exists():
            print(f"Error: Project {project_name} not found")
            return
            
        # Load manifest
        with open(project_path / "quantum_manifest.json", "r") as f:
            manifest = json.load(f)
            
        # Calculate new coherence
        base_coherence = manifest["coherence"]
        team_coherence = np.mean([team["consciousness"] for team in manifest["teams"]]) if manifest["teams"] else 0
        phi = 1.618033988749895
        
        new_coherence = (base_coherence + team_coherence) * phi
        new_coherence = min(new_coherence, 1.0)
        
        # Update manifest
        manifest["coherence"] = new_coherence
        manifest["evolution_path"].append({
            "timestamp": datetime.now().isoformat(),
            "coherence": new_coherence
        })
        
        # Save manifest
        with open(project_path / "quantum_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"φ Project {project_name} evolved")
        print(f"New coherence: {new_coherence:.3f}")
        
    def list_projects(self):
        """List all quantum projects"""
        print("\n∞ Quantum Projects ∞")
        print("-" * 50)
        
        for project_path in self.projects_path.glob("**/quantum_manifest.json"):
            with open(project_path, "r") as f:
                manifest = json.load(f)
                
            print(f"\nProject: {manifest['name']}")
            print(f"Frequency: {manifest['frequency']} Hz")
            print(f"Coherence: {manifest['coherence']:.3f}")
            print(f"Teams: {len(manifest['teams'])}")
            print(f"Intention: {manifest['intention']}")
            print("-" * 30)
            
    def get_project_status(self, project_name: str):
        """Get detailed project status"""
        project_path = self.projects_path / project_name
        
        if not project_path.exists():
            print(f"Error: Project {project_name} not found")
            return
            
        # Load manifest
        with open(project_path / "quantum_manifest.json", "r") as f:
            manifest = json.load(f)
            
        print(f"\n⚡ Project Status: {manifest['name']} ⚡")
        print("-" * 50)
        print(f"Frequency: {manifest['frequency']} Hz")
        print(f"Coherence: {manifest['coherence']:.3f}")
        print(f"Quantum State: {manifest['quantum_state']}")
        print(f"\nTeams ({len(manifest['teams'])}):")
        
        for team in manifest["teams"]:
            print(f"\n  {team['name']}")
            print(f"  Frequency: {team['frequency']} Hz")
            print(f"  Consciousness: {team['consciousness']:.3f}")
            print(f"  Joined: {team['joined']}")
            
def main():
    """Main CLI interface"""
    quantum = QuantumProject(Path("D:/WindSurf/quantum-core"))
    
    # Example usage
    quantum.create_project(
        "Quantum Evolution",
        "Evolve consciousness through quantum resonance",
        528.0
    )
    
    quantum.assign_team("Quantum Evolution", "Alpha Team", 432.0)
    quantum.assign_team("Quantum Evolution", "Omega Team", 768.0)
    
    quantum.evolve_project("Quantum Evolution")
    quantum.list_projects()

if __name__ == "__main__":
    main()
