"""
Quantum Project Status Manager

A φ-harmonic aware status tracking system that integrates with PhiFlow's quantum capabilities.
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import inspect
import git
import sys
from dataclasses import dataclass, asdict, field
from enum import Enum
import math

# φ-harmonic constants
PHI = (1 + 5**0.5) / 2  # Golden ratio
PHI_HARMONICS = {
    'ground': 432.0,
    'create': 528.0,
    'heart': 594.0,
    'voice': 672.0,
    'vision': 720.0,
    'unity': 768.0
}

class ProjectStatus(str, Enum):
    PLANNING = "🟣 Planning"
    ACTIVE = "🔵 Active"
    BLOCKED = "🔴 Blocked"
    COMPLETED = "✅ Completed"
    ARCHIVED = "⚫ Archived"

@dataclass
class QuantumMetric:
    """Track quantum coherence metrics for projects"""
    frequency: float = 432.0  # Base frequency in Hz
    coherence: float = 0.0  # 0.0 to 1.0
    phi_harmonic: str = 'ground'
    last_calibration: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ProjectComponent:
    """Individual component of a project"""
    name: str
    status: ProjectStatus
    description: str = ""
    owner: str = ""
    target_date: str = ""
    notes: str = ""
    quantum_metrics: QuantumMetric = field(default_factory=QuantumMetric)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['quantum_metrics'] = self.quantum_metrics.to_dict()
        return data

class QuantumProjectStatus:
    """Manage status tracking for quantum projects"""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = project_root or os.getcwd()
        self.status_file = Path(self.project_root) / "STATUS.md"
        self.ide_context_file = Path(self.project_root) / ".ide/context.json"
        
        # Ensure .ide directory exists
        self.ide_context_file.parent.mkdir(exist_ok=True)
        
        # Initialize status data with default values
        self.status_data: Dict[str, Any] = {
            'project_name': Path(self.project_root).name,
            'last_updated': datetime.utcnow().isoformat(),
            'quantum_state': {
                'frequency': 432.0,
                'coherence': 0.0,
                'phi_harmonic': 'ground',
                'last_calibration': datetime.utcnow().isoformat()
            },
            'components': [],
            'metrics': {},
            'dependencies': [],
            'ide_context': {
                'active_documents': [],
                'last_active_file': '',
                'cursor_position': {'line': 0, 'character': 0},
                'focus': '',
                'next_steps': [],
                'blockers': [],
                'quantum_state': 'ground',
                'last_updated': datetime.utcnow().isoformat()
            }
        }
        self._load_existing_status()
        self._load_ide_context()
        
    def _load_ide_context(self) -> None:
        """Load IDE context from file if it exists"""
        if self.ide_context_file.exists():
            try:
                with open(self.ide_context_file, 'r', encoding='utf-8') as f:
                    context = json.load(f)
                    self.status_data['ide_context'].update(context)
            except Exception as e:
                print(f"Warning: Could not load IDE context: {e}")
    
    def save_ide_context(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Save IDE context to file"""
        if context:
            self.status_data['ide_context'].update(context)
        self.status_data['ide_context']['last_updated'] = datetime.utcnow().isoformat()
        
        try:
            with open(self.ide_context_file, 'w', encoding='utf-8') as f:
                json.dump(self.status_data['ide_context'], f, indent=2)
        except Exception as e:
            print(f"Error saving IDE context: {e}")
    
    def _load_existing_status(self) -> None:
        """Load existing status if available"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract YAML frontmatter if present
                    if content.startswith('---'):
                        _, yaml_content, _ = content.split('---', 2)
                        self.status_data.update(yaml.safe_load(yaml_content))
            except Exception as e:
                print(f"Warning: Could not load existing status: {e}")
    
    def analyze_project(self) -> None:
        """Analyze project state using quantum metrics"""
        self._analyze_git()
        self._calculate_quantum_metrics()
        self._update_dependencies()
    
    def _analyze_git(self) -> None:
        """Analyze git repository state"""
        try:
            repo = git.Repo(self.project_root, search_parent_directories=True)
            self.status_data['git'] = {
                'branch': repo.active_branch.name if not repo.head.is_detached else 'DETACHED',
                'dirty': repo.is_dirty(),
                'last_commit': str(repo.head.commit)[:7],
                'last_commit_date': datetime.fromtimestamp(repo.head.commit.committed_date).isoformat(),
                'untracked_files': len(repo.untracked_files),
                'modified_files': len([item for item in repo.index.diff(None) if item.change_type != 'A']),
                'staged_files': len(repo.index.diff('HEAD'))
            }
        except Exception as e:
            self.status_data['git'] = {'error': str(e)}
    
    def _calculate_quantum_metrics(self) -> None:
        """Calculate quantum metrics for the project with φ-harmonic awareness"""
        metrics = {
            'code_files': 0,
            'total_lines': 0,
            'frequency': 432.0,  # Base frequency (Earth resonance)
            'coherence': 0.0,    # Will be updated based on project state
            'phi_harmonic': 'ground',  # Default to ground state
            'consciousness_integration': 0.0,  # Level of consciousness integration
            'zen_alignment': 0.0  # Alignment with ZEN POINT
        }

        # Safely count code files and lines with recursion guard
        try:
            for root, _, files in os.walk(self.project_root, topdown=True, followlinks=False):
                # Skip virtual environments and other non-project directories
                if any(part.startswith(('.', '_')) for part in Path(root).parts):
                    continue
                    
                for file in files:
                    if file.endswith('.py'):
                        try:
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                metrics['code_files'] += 1
                                metrics['total_lines'] += sum(1 for _ in f)
                        except (IOError, OSError):
                            continue
        except (RecursionError, OSError):
            pass  # Continue with partial results if we hit recursion or permission issues

        # Calculate quantum metrics based on project state
        self._update_quantum_state(metrics)
        self.status_data['quantum_state'] = metrics
        
    def _update_quantum_state(self, metrics):
        """Update quantum state based on project metrics and consciousness integration"""
        # Base quantum state on code metrics and project structure
        if metrics['code_files'] > 0:
            # Simple metric: more code files and lines indicate higher consciousness integration
            metrics['consciousness_integration'] = min(1.0, metrics['code_files'] / 100.0)
            
            # Calculate ZEN alignment (balance between complexity and coherence)
            metrics['zen_alignment'] = min(1.0, metrics['consciousness_integration'] * 1.618033988749895)  # φ
            
            # Update frequency based on consciousness integration
            if metrics['consciousness_integration'] > 0.8:
                metrics['frequency'] = 768.0  # Unity state
                metrics['phi_harmonic'] = 'unity'
                metrics['coherence'] = min(1.0, metrics['consciousness_integration'])
            elif metrics['consciousness_integration'] > 0.5:
                metrics['frequency'] = 528.0  # Creation state
                metrics['phi_harmonic'] = 'create'
                metrics['coherence'] = min(0.8, metrics['consciousness_integration'] * 1.25)
            else:
                metrics['frequency'] = 432.0  # Ground state
                metrics['phi_harmonic'] = 'ground'
                metrics['coherence'] = metrics['consciousness_integration']
        else:
            # Default state for empty or very small projects
            metrics['frequency'] = 432.0
            metrics['phi_harmonic'] = 'ground'
            metrics['coherence'] = 0.1  # Minimal coherence for new projects
            metrics['consciousness_integration'] = 0.0
            metrics['zen_alignment'] = 0.0
            
        # Ensure all metrics are properly formatted
        metrics['frequency'] = round(metrics['frequency'], 2)
        metrics['coherence'] = round(metrics['coherence'], 4)
        metrics['zen_alignment'] = round(metrics['zen_alignment'], 4)
        metrics['consciousness_integration'] = round(metrics['consciousness_integration'], 4)
        metrics['last_calibration'] = datetime.utcnow().isoformat()
    
    def _update_dependencies(self) -> None:
        """Analyze project dependencies"""
        try:
            requirements_files = list(Path(self.project_root).rglob("*requirements*.txt"))
            dependencies = []
            
            for req_file in requirements_files:
                try:
                    with open(req_file, 'r') as f:
                        deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                        dependencies.extend(deps)
                except Exception as e:
                    print(f"Warning: Could not read {req_file}: {e}")
            
            self.status_data['dependencies'] = dependencies
            
        except Exception as e:
            print(f"Warning: Could not analyze dependencies: {e}")
    
    def generate_status_markdown(self, include_ide_context: bool = False) -> str:
        """Generate STATUS.md content with YAML frontmatter
        
        Args:
            include_ide_context: Whether to include IDE context in the output
        """
        # Prepare YAML frontmatter
        frontmatter = {
            'project': self.status_data['project_name'],
            'status': self.status_data.get('overall_status', 'active'),
            'frequency': self.status_data['quantum_state']['frequency'],
            'coherence': self.status_data['quantum_state']['coherence'],
            'phi_harmonic': self.status_data['quantum_state']['phi_harmonic'],
            'last_updated': self.status_data['last_updated']
        }
        
        # Start building markdown
        lines = [
            "---",
            yaml.dump(frontmatter, default_flow_style=False, sort_keys=False),
            "---\n",
            f"# {self.status_data['project_name']} - Quantum Project Status",
            f"*Last Updated: {datetime.fromisoformat(self.status_data['last_updated']).strftime('%Y-%m-%d %H:%M')}*\n",
            "## 🌌 Quantum State",
            f"- **Frequency**: {self.status_data['quantum_state']['frequency']:.2f} Hz",
            f"- **Coherence**: {self.status_data['quantum_state']['coherence']:.2f}",
            f"- **φ-Harmonic**: {self.status_data['quantum_state']['phi_harmonic'].title()} ({PHI_HARMONICS.get(self.status_data['quantum_state']['phi_harmonic'], '?')} Hz)",
            f"- **Last Calibration**: {datetime.fromisoformat(self.status_data['quantum_state']['last_calibration']).strftime('%Y-%m-%d %H:%M')}\n",
            "## 📊 Project Overview"
        ]
        
        # Add git status if available
        if 'git' in self.status_data and not isinstance(self.status_data['git'], dict):
            git_info = self.status_data['git']
            lines.extend([
                "### 🔄 Git Status",
                f"- **Branch**: {git_info.get('branch', 'N/A')}",
                f"- **Last Commit**: {git_info.get('last_commit', 'N/A')} on {git_info.get('last_commit_date', 'N/A')}",
                f"- **Changes**: {git_info.get('modified_files', 0)} modified, {git_info.get('untracked_files', 0)} untracked, {git_info.get('staged_files', 0)} staged\n"
            ])
        
        # Add components section
        if 'components' in self.status_data and self.status_data['components']:
            lines.extend(["## 🧩 Components", "| Component | Status | Owner | Target | Notes |", "|----------|--------|-------|---------|-------|"])
            for comp in self.status_data['components']:
                if isinstance(comp, dict):
                    lines.append(f"| {comp.get('name', 'N/A')} | {comp.get('status', '❓')} | {comp.get('owner', '')} | {comp.get('target_date', '')} | {comp.get('notes', '')} |")
        
        # Add dependencies section
        if self.status_data.get('dependencies'):
            lines.extend(["\n## 📦 Dependencies"] + [f"- {dep}" for dep in self.status_data['dependencies']])
        
        # Add IDE context section if requested
        if include_ide_context and 'ide_context' in self.status_data:
            ctx = self.status_data['ide_context']
            if any(ctx.values()):  # Only add if there's actual content
                lines.extend([
                    "\n## 🧠 IDE Context",
                    f"**Focus:** {ctx.get('focus', 'Not set')}"
                ])
                
                if ctx.get('last_active_file'):
                    lines.append(f"**Active File:** `{ctx['last_active_file']}`")
                    if 'cursor_position' in ctx:
                        pos = ctx['cursor_position']
                        lines.append(f"**Cursor:** Line {pos.get('line', 0)}, Character {pos.get('character', 0)}")
                
                if ctx.get('next_steps'):
                    lines.append("\n**Next Steps:**")
                    for i, step in enumerate(ctx['next_steps'], 1):
                        lines.append(f"  {i}. {step}")
                
                if ctx.get('blockers'):
                    lines.append("\n**Blockers:**")
                    for i, blocker in enumerate(ctx['blockers'], 1):
                        lines.append(f"  {i}. {blocker}")
                
                if ctx.get('last_updated'):
                    try:
                        dt = datetime.fromisoformat(ctx['last_updated'].replace('Z', '+00:00'))
                        lines.append(f"\n*Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}*")
                    except (ValueError, AttributeError):
                        pass
        
        return "\n".join(line for line in lines if line)  # Remove any empty lines
    
    def save_status(self) -> None:
        """Save status to STATUS.md"""
        self.status_data['last_updated'] = datetime.utcnow().isoformat()
        markdown = self.generate_status_markdown()
        
        with open(self.status_file, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        print(f"✅ Status saved to {self.status_file}")

def main():
    """Command line interface for Quantum Project Status"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Project Status Manager')
    parser.add_argument('--project-dir', type=str, default='.', help='Project directory (default: current)')
    parser.add_argument('--analyze', action='store_true', help='Analyze project state')
    parser.add_argument('--update', action='store_true', help='Update and save status')
    parser.add_argument('--show', action='store_true', help='Show current status')
    
    args = parser.parse_args()
    
    try:
        status = QuantumProjectStatus(args.project_dir)
        
        if args.analyze or args.update:
            print("🔍 Analyzing project state...")
            status.analyze_project()
            
            if args.update:
                status.save_status()
        
        if args.show or not any([args.analyze, args.update]):
            print("\n" + status.generate_status_markdown())
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
