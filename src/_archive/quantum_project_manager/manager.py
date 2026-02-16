import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path # Added Path import

from .project_model import QuantumProject
from .project_store import ProjectStore

# It's good practice for a library module to use getLogger(__name__)
# and let the application configure handlers. Adding a NullHandler can prevent
# "No handler found" warnings if the application doesn't configure logging.
manager_logger = logging.getLogger(__name__)
manager_logger.addHandler(logging.NullHandler()) # Add NullHandler to avoid warnings if not configured by app

class QuantumProjectManager:
    def __init__(self, storage_path: str = "d:/Projects/PhiFlow/data/quantum_projects"):
        self.store = ProjectStore(storage_path_str=storage_path)
        manager_logger.info(f"QuantumProjectManager initialized. Storage path: {self.store.storage_path.resolve()}")

    def _log_to_greg_quantum_log(self, message: str):
        greg_log_path = "D:\\Greg\\Quantum_Log.txt"
        try:
            # Ensure directory exists (though it should if CymaticsModule ran)
            Path(greg_log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(greg_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                f.write(f"{timestamp} - QuantumProjectManager - INFO - {message}\n")
        except Exception as e:
            manager_logger.error(f"Failed to write to Greg's Quantum Log ({greg_log_path}): {e}")

    def create_project(self, name: str, creator: str = "Greg", **kwargs: Any) -> QuantumProject:
        project = QuantumProject(name=name, creator=creator, **kwargs)
        self.store.save_project(project)
        log_msg = f"Project CREATED: Name='{project.name}', ID='{project.project_id}', Creator='{project.creator}'"
        manager_logger.info(log_msg)
        self._log_to_greg_quantum_log(log_msg)
        return project

    def get_project(self, project_id: str) -> Optional[QuantumProject]:
        project = self.store.load_project(project_id)
        if project:
            manager_logger.info(f"Project RETRIEVED: Name='{project.name}', ID='{project.project_id}'")
        else:
            manager_logger.warning(f"Project with ID '{project_id}' not found for retrieval.")
        return project

    def update_project(self, project_id: str, updates: Dict[str, Any]) -> Optional[QuantumProject]:
        project = self.store.load_project(project_id)
        if not project:
            manager_logger.warning(f"Update failed: Project with ID '{project_id}' not found.")
            return None

        updated_fields = []
        for key, value in updates.items():
            if hasattr(project, key):
                setattr(project, key, value)
                updated_fields.append(key)
            else:
                manager_logger.warning(f"Update for project '{project.name}' (ID: {project_id}): Invalid field '{key}' ignored.")
        
        if not updated_fields:
            manager_logger.info(f"No valid fields to update for project '{project.name}' (ID: {project_id}).")
            # Still, we might want to update 'last_updated_date' if any attempt was made
            # For now, only save if actual fields were changed.
            # return project # Return the unmodified project

        project.last_updated_date = datetime.now() # Ensure this is updated
        self.store.save_project(project)
        log_msg = f"Project UPDATED: Name='{project.name}', ID='{project.project_id}', Fields changed: {', '.join(updated_fields) if updated_fields else 'None'}"
        manager_logger.info(log_msg)
        self._log_to_greg_quantum_log(log_msg)
        return project

    def delete_project(self, project_id: str) -> bool:
        project_to_log = self.store.load_project(project_id) # Load for logging name before deletion
        deleted = self.store.delete_project(project_id)
        if deleted and project_to_log:
            log_msg = f"Project DELETED: Name='{project_to_log.name}', ID='{project_id}'"
            manager_logger.info(log_msg)
            self._log_to_greg_quantum_log(log_msg)
            return True
        elif deleted: # Should ideally not happen if project_id was valid and load failed
            log_msg = f"Project DELETED: ID='{project_id}' (Name not available for logging)"
            manager_logger.info(log_msg)
            self._log_to_greg_quantum_log(log_msg)
            return True
        else:
            manager_logger.warning(f"Delete failed: Project with ID '{project_id}' not found.")
            return False

    def list_projects(self, status_filter: Optional[str] = None, creator_filter: Optional[str] = None, name_contains: Optional[str] = None) -> List[QuantumProject]:
        all_projects = self.store.load_all_projects()
        
        if not all_projects:
            manager_logger.info("No projects found in the store.")
            return []
            
        filtered_projects = all_projects
        
        if status_filter:
            filtered_projects = [p for p in filtered_projects if p.status == status_filter]
        if creator_filter:
            filtered_projects = [p for p in filtered_projects if p.creator == creator_filter]
        if name_contains:
            filtered_projects = [p for p in filtered_projects if name_contains.lower() in p.name.lower()]
        
        manager_logger.info(f"Listed {len(filtered_projects)} of {len(all_projects)} total projects (Filters: Status='{status_filter}', Creator='{creator_filter}', NameContains='{name_contains}')")
        return filtered_projects
