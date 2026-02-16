#!/usr/bin/env python3
"""
Quantum Project Status Manager (qstatus)

A φ-harmonic aware status tracking system for quantum projects.
"""

import os
import sys
import click
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_project_manager.status_manager import QuantumProjectStatus, ProjectStatus, ProjectComponent, QuantumMetric

@click.group()
@click.version_option("1.0.0", message="Quantum Status Manager %(version)s")
@click.pass_context
def cli(ctx):
    """Quantum Project Status Manager - Track and manage project status with φ-harmonic awareness"""
    ctx.ensure_object(dict)
    ctx.obj['project_root'] = os.getcwd()

def _get_harmonic_emoji(harmonic_name: str) -> str:
    """Get emoji for harmonic state"""
    emoji_map = {
        'ground': '🌍',  # Earth resonance
        'create': '✨',  # Creation/transformation
        'heart': '💖',  # Heart coherence
        'voice': '🎵',  # Sound/expression
        'vision': '👁️',  # Perception
        'unity': '🌀'   # Unity/wholeness
    }
    return emoji_map.get(harmonic_name.lower(), '⚡')

@cli.command()
@click.argument('focus')
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
@click.option('--next-steps', '-n', multiple=True, help='Next steps (can specify multiple)')
@click.option('--blockers', '-b', multiple=True, help='Current blockers (can specify multiple)')
@click.option('--quantum-state', '-q', 
              type=click.Choice(['ground', 'create', 'heart', 'voice', 'vision', 'unity']),
              help='Current quantum state')
@click.option('--active-file', '-f', help='Currently active file')
@click.option('--line', '-l', type=int, help='Cursor line number')
@click.option('--character', '-c', type=int, help='Cursor character position')
def context(focus: str, project_dir: str, next_steps: list, blockers: list, 
           quantum_state: str, active_file: str, line: int, character: int):
    """Update IDE context with current focus and state"""
    status = QuantumProjectStatus(project_dir)
    
    context_update = {
        'focus': focus,
        'next_steps': list(next_steps) if next_steps else [],
        'blockers': list(blockers) if blockers else [],
        'last_updated': datetime.utcnow().isoformat()
    }
    
    if quantum_state:
        context_update['quantum_state'] = quantum_state
    
    if active_file:
        context_update['last_active_file'] = active_file
        
        # Update cursor position if provided
        if line is not None and character is not None:
            context_update['cursor_position'] = {
                'line': line,
                'character': character
            }
    
    status.save_ide_context(context_update)
    click.echo(f"✓ Updated IDE context: {focus}")

@cli.command()
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
@click.option('--full/--brief', default=False, help='Show full context including file and position')
def resume(project_dir: str, full: bool):
    """Resume work from last saved context"""
    status = QuantumProjectStatus(project_dir)
    context = status.status_data.get('ide_context', {})
    
    if not context.get('focus'):
        click.echo("No saved context found. Use 'context' command to set your focus.")
        return
    
    click.echo(f"📌 Last Focus: {context.get('focus', 'N/A')}")
    
    if full:
        if context.get('last_active_file'):
            click.echo(f"📄 File: {context.get('last_active_file')}")
            if 'cursor_position' in context:
                pos = context['cursor_position']
                click.echo(f"   At line {pos.get('line', 0)}, character {pos.get('character', 0)}")
    
    if context.get('next_steps'):
        click.echo("\n➡️ Next Steps:")
        for i, step in enumerate(context['next_steps'], 1):
            click.echo(f"  {i}. {step}")
    
    if context.get('blockers'):
        click.echo("\n⚠️  Blockers:")
        for i, blocker in enumerate(context['blockers'], 1):
            click.echo(f"  {i}. {blocker}")
    
    quantum_state = context.get('quantum_state', 'ground')
    emoji = _get_harmonic_emoji(quantum_state)
    click.echo(f"\n🌌 Quantum State: {quantum_state.title()} {emoji}")
    
    last_updated = context.get('last_updated', '')
    if last_updated:
        try:
            dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            click.echo(f"\n⏱️  Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except (ValueError, AttributeError):
            pass

@cli.command()
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
@click.option('--output', '-o', type=click.Choice(['markdown', 'json', 'yaml']), default='markdown',
              help='Output format')
@click.option('--save/--no-save', default=False, help='Save status to STATUS.md')
@click.option('--ide-context', is_flag=True, help='Include IDE context in output')
def show(project_dir: str, output: str, save: bool, ide_context: bool):
    """Show current project status"""
    status = QuantumProjectStatus(project_dir)
    status.analyze_project()
    
    if save:
        status.save_status()
    
    # Prepare output data
    output_data = status.status_data.copy()
    
    # Only include IDE context if explicitly requested
    if not ide_context and 'ide_context' in output_data:
        del output_data['ide_context']
        
    if output == 'json':
        import json
        print(json.dumps(output_data, indent=2))
    elif output == 'yaml':
        import yaml
        print(yaml.dump(output_data, default_flow_style=False))
    else:  # markdown
        print(status.generate_status_markdown(include_ide_context=ide_context))
        
        # Get quantum state data
        quantum_state = status.status_data.get('quantum_state', {})
        
        # Update the existing quantum state section with enhanced formatting
        if '## 🌌 Quantum State' in output_data:
            # Remove existing quantum state section
            md_parts = md.split('## 🌌 Quantum State', 1)
            before_quantum = md_parts[0]
            after_quantum = md_parts[1].split('##', 1)[1] if '##' in md_parts[1] else ''
            
            # Create enhanced quantum section
            quantum_section = [
                "## 🌌 Quantum State",
                f"- **Frequency**: {quantum_state.get('frequency', 432.0):.2f} Hz {_get_harmonic_emoji(quantum_state.get('phi_harmonic', 'ground'))} ({quantum_state.get('phi_harmonic', 'ground').title()})",
                f"- **Coherence**: {quantum_state.get('coherence', 0.0):.2f}/1.0",
                f"- **ZEN Alignment**: {quantum_state.get('zen_alignment', 0.0):.2f}φ",
                f"- **Consciousness Integration**: {quantum_state.get('consciousness_integration', 0.0):.0%}",
            ]
            
            # Add last calibration time
            last_cal = quantum_state.get('last_calibration', datetime.utcnow().isoformat())
            try:
                cal_time = datetime.fromisoformat(last_cal.replace('Z', '+00:00'))
                quantum_section.append(f"- **Last Calibration**: {cal_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            except (ValueError, AttributeError):
                quantum_section.append(f"- **Last Calibration**: {last_cal}")
            
            # Rebuild the markdown
            md = f"{before_quantum}\n\n" + '\n'.join(quantum_section)
            if after_quantum:
                md += f"\n\n##{after_quantum}"
        
        print("\n" + md)

@cli.command()
@click.argument('name')
@click.option('--status', '-s', 
              type=click.Choice(['planning', 'active', 'blocked', 'completed', 'archived'], 
                              case_sensitive=False),
              default='active')
@click.option('--owner', '-o', default='', help='Component owner')
@click.option('--target-date', '-d', help='Target completion date (YYYY-MM-DD)')
@click.option('--notes', '-n', default='', help='Additional notes')
@click.option('--frequency', '-f', type=float, help='Target frequency in Hz')
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
def add_component(name: str, status: str, owner: str, target_date: str, 
                 notes: str, frequency: Optional[float], project_dir: str):
    """Add a new component to track"""
    status_obj = QuantumProjectStatus(project_dir)
    
    # Get existing components
    components = status_obj.status_data.get('components', [])
    
    # Create quantum metric
    quantum_metric = QuantumMetric()
    if frequency:
        quantum_metric.frequency = frequency
        # Find nearest φ-harmonic
        quantum_metric.phi_harmonic = min(
            status_obj.status_data['quantum_state'].get('phi_harmonics', {}).items(),
            key=lambda x: abs(x[1] - frequency)
        )[0] if status_obj.status_data['quantum_state'].get('phi_harmonics') else 'ground'
    
    # Create new component
    new_component = {
        'name': name,
        'status': getattr(ProjectStatus, status.upper()).value,
        'owner': owner,
        'target_date': target_date or '',
        'notes': notes,
        'quantum_metrics': quantum_metric.to_dict()
    }
    
    components.append(new_component)
    status_obj.status_data['components'] = components
    status_obj.save_status()
    
    click.echo(f"✅ Added component: {name}")

@cli.command()
@click.argument('component_name')
@click.option('--status', '-s', 
              type=click.Choice(['planning', 'active', 'blocked', 'completed', 'archived'], 
                              case_sensitive=False))
@click.option('--owner', '-o', help='Component owner')
@click.option('--target-date', '-d', help='Target completion date (YYYY-MM-DD)')
@click.option('--notes', '-n', help='Additional notes')
@click.option('--frequency', '-f', type=float, help='Target frequency in Hz')
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
def update_component(component_name: str, status: Optional[str], owner: Optional[str], 
                   target_date: Optional[str], notes: Optional[str], 
                   frequency: Optional[float], project_dir: str):
    """Update an existing component"""
    status_obj = QuantumProjectStatus(project_dir)
    components = status_obj.status_data.get('components', [])
    
    found = False
    for comp in components:
        if isinstance(comp, dict) and comp.get('name') == component_name:
            found = True
            if status:
                comp['status'] = getattr(ProjectStatus, status.upper()).value
            if owner is not None:
                comp['owner'] = owner
            if target_date is not None:
                comp['target_date'] = target_date
            if notes is not None:
                comp['notes'] = notes
            if frequency is not None:
                if 'quantum_metrics' not in comp:
                    comp['quantum_metrics'] = {}
                comp['quantum_metrics']['frequency'] = frequency
    
    if not found:
        click.echo(f"❌ Component '{component_name}' not found", err=True)
        return
    
    status_obj.status_data['components'] = components
    status_obj.save_status()
    click.echo(f"✅ Updated component: {component_name}")

@cli.command()
@click.argument('component_name')
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
@click.confirmation_option(prompt='Are you sure you want to remove this component?')
def remove_component(component_name: str, project_dir: str):
    """Remove a component from tracking"""
    status_obj = QuantumProjectStatus(project_dir)
    components = status_obj.status_data.get('components', [])
    
    initial_count = len(components)
    components = [c for c in components if not (isinstance(c, dict) and c.get('name') == component_name)]
    
    if len(components) == initial_count:
        click.echo(f"❌ Component '{component_name}' not found", err=True)
        return
    
    status_obj.status_data['components'] = components
    status_obj.save_status()
    click.echo(f"✅ Removed component: {component_name}")

@cli.command()
@click.option('--frequency', '-f', type=float, default=432.0, 
              help='Base frequency in Hz (default: 432.0)')
@click.option('--coherence', '-c', type=float, default=0.0,
              help='Coherence level (0.0 to 1.0)')
@click.option('--harmonic', '-h', 
              type=click.Choice(['ground', 'create', 'heart', 'voice', 'vision', 'unity'], 
                              case_sensitive=False),
              help='φ-Harmonic state')
@click.option('--project-dir', '-p', default='.', help='Project directory (default: current)')
def calibrate(frequency: float, coherence: float, harmonic: Optional[str], project_dir: str):
    """Calibrate quantum state"""
    status_obj = QuantumProjectStatus(project_dir)
    
    if 'quantum_state' not in status_obj.status_data:
        status_obj.status_data['quantum_state'] = {}
    
    if frequency:
        status_obj.status_data['quantum_state']['frequency'] = frequency
    if coherence is not None:
        status_obj.status_data['quantum_state']['coherence'] = max(0.0, min(1.0, coherence))
    if harmonic:
        status_obj.status_data['quantum_state']['phi_harmonic'] = harmonic
    
    status_obj.status_data['quantum_state']['last_calibration'] = datetime.utcnow().isoformat()
    status_obj.save_status()
    
    click.echo("🔮 Quantum state calibrated:")
    click.echo(f"   Frequency: {status_obj.status_data['quantum_state']['frequency']} Hz")
    click.echo(f"   Coherence: {status_obj.status_data['quantum_state']['coherence']:.2f}")
    click.echo(f"   φ-Harmonic: {status_obj.status_data['quantum_state'].get('phi_harmonic', 'ground').title()}")

if __name__ == "__main__":
    cli()
