import argparse
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import json # For pretty printing project details

# --- Path Setup ---
# Adjust Python path to include the 'src' directory for direct script execution
# This allows importing 'quantum_project_manager'
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent # Should be d:\Projects\PhiFlow\src

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# --- Imports from Quantum Project Manager ---
try:
    from quantum_project_manager import QuantumProjectManager, QuantumProject
except ImportError as e:
    print(f"❌ Error importing QuantumProjectManager or QuantumProject: {e}")
    print(f"Ensure 'quantum_project_manager' is in PYTHONPATH or script is run from 'src' parent directory.")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted to add SRC_DIR: {SRC_DIR}")
    sys.exit(1)

# --- Logging Setup ---
cli_logger = logging.getLogger(__name__)
# BasicConfig for the CLI application itself.
# The format will be distinct for CLI messages vs manager messages.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - QPM_CLI - %(levelname)s - %(message)s')

# Get the logger used by QuantumProjectManager to control its verbosity via CLI flag
qpm_internal_logger = logging.getLogger('quantum_project_manager.manager')
# Default to WARNING to keep manager logs quiet unless --verbose is used.
# The manager itself adds a NullHandler, so it won't complain if no handlers are set here.
qpm_internal_logger.setLevel(logging.WARNING)

# --- Helper Functions ---
def print_project_details(project: QuantumProject, title: str = "Project Details"):
    """Prints project details in a readable JSON format."""
    if project:
        print(f"\n=== {title.upper()} ===")
        # Convert to dict for consistent serialization of datetime
        project_data = project.to_dict()
        print(json.dumps(project_data, indent=2, ensure_ascii=False))
        print("\n")
    else:
        print("No project data to display.")

def parse_datetime_string(dt_string: Optional[str]) -> Optional[datetime]:
    if not dt_string:
        return None
    try:
        return datetime.fromisoformat(dt_string)
    except ValueError:
        cli_logger.warning(f"Invalid datetime format: '{dt_string}'. Please use ISO format (YYYY-MM-DDTHH:MM:SS).")
        return None # Or raise error depending on strictness

# --- Command Handler Functions ---
def handle_create(args, manager: QuantumProjectManager):
    cli_logger.info(f"Attempting to create project: {args.name}")
    project_data = {
        "name": args.name,
        "objective": args.objective,
        "status": args.status,
        "creator": args.creator,
        "primary_frequency_hz": args.primary_frequency_hz,
        "target_coherence": args.target_coherence,
        "associated_harmonics": args.associated_harmonics or [],
        "quantum_components_used": args.quantum_components_used or [],
        "wizdome_era": args.wizdome_era,
        "being_phi_state": args.being_phi_state,
        "target_completion_date": parse_datetime_string(args.target_completion_date),
        "associated_files_paths": args.associated_files_paths or [],
        "experiment_ids": args.experiment_ids or [],
        "know_link_references": args.know_link_references or [],
        "notes": args.notes
    }
    # Filter out None values so that model defaults can apply
    project_data_cleaned = {k: v for k, v in project_data.items() if v is not None}
    
    new_project = manager.create_project(**project_data_cleaned)
    if new_project:
        cli_logger.info(f"✅ Project '{new_project.name}' (ID: {new_project.project_id}) created successfully.")
        print_project_details(new_project, "Newly Created Project")
    else:
        cli_logger.error("❌ Failed to create project.")

def handle_get(args, manager: QuantumProjectManager):
    cli_logger.info(f"Retrieving project with ID: {args.project_id}")
    project = manager.get_project(args.project_id)
    if project:
        print_project_details(project, f"Details for Project ID: {args.project_id}")
    else:
        cli_logger.warning(f"⚠️ Project with ID '{args.project_id}' not found.")

def handle_list(args, manager: QuantumProjectManager):
    cli_logger.info(f"Listing projects... Filters: Status='{args.status}', Creator='{args.creator}', NameContains='{args.name_contains}'")
    projects = manager.list_projects(
        status_filter=args.status,
        creator_filter=args.creator,
        name_contains=args.name_contains
    )
    if projects:
        print(f"\n=== FOUND {len(projects)} PROJECT(S) ===")
        for i, p in enumerate(projects):
            print(f"  {i+1}. ID: {p.project_id}\n     Name: {p.name}\n     Status: {p.status}\n     Creator: {p.creator}\n     Primary Frequency: {p.primary_frequency_hz or 'N/A'} Hz")
            if args.verbose_list:
                 print(f"     Objective: {p.objective[:80] + '...' if p.objective and len(p.objective) > 80 else p.objective}")
                 print(f"     Last Updated: {p.last_updated_date.strftime('%Y-%m-%d %H:%M') if p.last_updated_date else 'N/A'}")
            print("     ----------")
        print("\n")
    else:
        cli_logger.info("ℹ️ No projects found matching the criteria.")

def handle_update(args, manager: QuantumProjectManager):
    cli_logger.info(f"Attempting to update project ID: {args.project_id}")
    updates = {}
    potential_updates = {
        "name": args.name,
        "objective": args.objective,
        "status": args.status,
        "creator": args.creator,
        "primary_frequency_hz": args.primary_frequency_hz,
        "target_coherence": args.target_coherence,
        "associated_harmonics": args.associated_harmonics,
        "quantum_components_used": args.quantum_components_used,
        "wizdome_era": args.wizdome_era,
        "being_phi_state": args.being_phi_state,
        "target_completion_date": parse_datetime_string(args.target_completion_date) if args.target_completion_date is not None else 'DO_NOT_UPDATE_FIELD_SENTINEL',
        "associated_files_paths": args.associated_files_paths,
        "experiment_ids": args.experiment_ids,
        "know_link_references": args.know_link_references,
        "notes": args.notes
    }
    for key, value in potential_updates.items():
        if value is not None and value != 'DO_NOT_UPDATE_FIELD_SENTINEL': # Check if arg was passed
            updates[key] = value
        elif key == 'target_completion_date' and value == 'DO_NOT_UPDATE_FIELD_SENTINEL' and args.target_completion_date is None and args.clear_target_completion_date:
            updates[key] = None # Explicitly set to None if --clear-target-completion-date is used
            
    if not updates:
        cli_logger.info("No update parameters provided. Nothing to do.")
        return

    updated_project = manager.update_project(args.project_id, updates)
    if updated_project:
        cli_logger.info(f"✅ Project '{updated_project.name}' (ID: {args.project_id}) updated successfully.")
        print_project_details(updated_project, "Updated Project Details")
    else:
        cli_logger.warning(f"⚠️ Failed to update project ID '{args.project_id}'. It might not exist or no valid fields were provided.")

def handle_delete(args, manager: QuantumProjectManager):
    cli_logger.info(f"Attempting to delete project ID: {args.project_id}")
    # Confirmation prompt
    if not args.yes:
        confirm = input(f"🚨 Are you sure you want to delete project ID '{args.project_id}'? This cannot be undone. (yes/no): ")
        if confirm.lower() != 'yes':
            cli_logger.info("Deletion cancelled by user.")
            return
            
    deleted = manager.delete_project(args.project_id)
    if deleted:
        cli_logger.info(f"✅ Project ID '{args.project_id}' deleted successfully.")
    else:
        cli_logger.warning(f"⚠️ Failed to delete project ID '{args.project_id}'. It might not exist.")

# --- Main CLI Function ---
def main():
    parser = argparse.ArgumentParser(description="Quantum Project Manager CLI 🌌 - Manage your quantum projects with vibrational precision.")
    parser.add_argument('--verbose', '-v', action='store_true', help="Enable verbose logging from the project manager internals.")
    parser.add_argument('--storage-path', type=str, default="d:/Projects/PhiFlow/data/quantum_projects", help="Override default storage path for project data.")
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- Create Command Parser ---
    create_parser = subparsers.add_parser('create', help='🌠 Create a new quantum project')
    create_parser.add_argument('--name', type=str, required=True, help='Name of the project (e.g., \"Project Phoenix\")')
    create_parser.add_argument('--objective', type=str, help='Objective of the project')
    create_parser.add_argument('--status', type=str, help='Initial status (e.g., Planning, ZEN_Incubation)')
    create_parser.add_argument('--creator', type=str, help='Creator of the project (default: Greg)')
    create_parser.add_argument('--primary-frequency-hz', type=float, help='Primary operational frequency in Hz (e.g., 432.0)')
    create_parser.add_argument('--target-coherence', type=str, help='Target coherence level (e.g., 0.95, \"φ\", \"φ^φ\")')
    create_parser.add_argument('--associated-harmonics', type=str, nargs='*', help='List of associated harmonics (e.g., \"528 Hz\" \"7.83 Hz\")')
    create_parser.add_argument('--quantum-components-used', type=str, nargs='*', help='List of quantum components (e.g., \"QuantumCore\" \"ZenPointField\")')
    create_parser.add_argument('--wizdome-era', type=str, help='WIZDOME Era (e.g., CASCADE_Era)')
    create_parser.add_argument('--being-phi-state', type=str, help='Being Phi State (e.g., Incubation)')
    create_parser.add_argument('--target-completion-date', type=str, help='Target completion date (ISO format: YYYY-MM-DDTHH:MM:SS)')
    create_parser.add_argument('--associated-files-paths', type=str, nargs='*', help='List of associated file paths')
    create_parser.add_argument('--experiment-ids', type=str, nargs='*', help='List of experiment IDs')
    create_parser.add_argument('--know-link-references', type=str, nargs='*', help='List of KNOW Link references')
    create_parser.add_argument('--notes', type=str, help='Free-form notes for the project')
    create_parser.set_defaults(func=handle_create)

    # --- Get Command Parser ---
    get_parser = subparsers.add_parser('get', help='🔍 Get details of a specific project')
    get_parser.add_argument('project_id', type=str, help='ID of the project to retrieve')
    get_parser.set_defaults(func=handle_get)

    # --- List Command Parser ---
    list_parser = subparsers.add_parser('list', help='📜 List all quantum projects')
    list_parser.add_argument('--status', type=str, help='Filter by project status')
    list_parser.add_argument('--creator', type=str, help='Filter by project creator')
    list_parser.add_argument('--name-contains', type=str, help='Filter by projects whose name contains the given text (case-insensitive)')
    list_parser.add_argument('--verbose-list', '-vl', action='store_true', help='Show more details in the list view.')
    list_parser.set_defaults(func=handle_list)

    # --- Update Command Parser ---
    update_parser = subparsers.add_parser('update', help='🛠️ Update an existing quantum project')
    update_parser.add_argument('project_id', type=str, help='ID of the project to update')
    update_parser.add_argument('--name', type=str, help='New name for the project')
    update_parser.add_argument('--objective', type=str, help='New objective')
    update_parser.add_argument('--status', type=str, help='New status')
    update_parser.add_argument('--creator', type=str, help='New creator')
    update_parser.add_argument('--primary-frequency-hz', type=float, help='New primary frequency')
    update_parser.add_argument('--target-coherence', type=str, help='New target coherence')
    update_parser.add_argument('--associated-harmonics', type=str, nargs='*', help='New list of associated harmonics (replaces existing)')
    update_parser.add_argument('--quantum-components-used', type=str, nargs='*', help='New list of quantum components (replaces existing)')
    update_parser.add_argument('--wizdome-era', type=str, help='New WIZDOME Era')
    update_parser.add_argument('--being-phi-state', type=str, help='New Being Phi State')
    update_parser.add_argument('--target-completion-date', type=str, help='New target completion date (ISO format). Use --clear-target-completion-date to remove.')
    update_parser.add_argument('--clear-target-completion-date', action='store_true', help='Set target_completion_date to None.')
    update_parser.add_argument('--associated-files-paths', type=str, nargs='*', help='New list of associated file paths (replaces existing)')
    update_parser.add_argument('--experiment-ids', type=str, nargs='*', help='New list of experiment IDs (replaces existing)')
    update_parser.add_argument('--know-link-references', type=str, nargs='*', help='New list of KNOW Link references (replaces existing)')
    update_parser.add_argument('--notes', type=str, help='New notes (replaces existing)')
    update_parser.set_defaults(func=handle_update)

    # --- Delete Command Parser ---
    delete_parser = subparsers.add_parser('delete', help='🗑️ Delete a quantum project')
    delete_parser.add_argument('project_id', type=str, help='ID of the project to delete')
    delete_parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt for deletion.')
    delete_parser.set_defaults(func=handle_delete)

    args = parser.parse_args()

    if args.verbose:
        qpm_internal_logger.setLevel(logging.INFO) # Set QPM manager logger to INFO if verbose
        cli_logger.info("Verbose logging enabled for project manager internals.")
    
    # Initialize the manager with potentially overridden storage path
    try:
        manager = QuantumProjectManager(storage_path=args.storage_path)
        cli_logger.info(f"Quantum Project Manager CLI initialized. Using storage: {manager.store.storage_path.resolve()}")
    except Exception as e:
        cli_logger.error(f"❌ Failed to initialize QuantumProjectManager: {e}")
        sys.exit(1)

    # Call the respective handler function for the subcommand
    args.func(args, manager)

if __name__ == "__main__":
    main()
