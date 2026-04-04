import json
import os
from pathlib import Path
from typing import Optional, List
from datetime import datetime # Ensure datetime is imported

from .project_model import QuantumProject

class ProjectStore:
    def __init__(self, storage_path_str: str = "d:/Projects/PhiFlow/data/quantum_projects"):
        self.storage_path = Path(storage_path_str)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_project_file_path(self, project_id: str) -> Path:
        return self.storage_path / f"{project_id}.json"

    def save_project(self, project: QuantumProject):
        file_path = self._get_project_file_path(project.project_id)
        project.last_updated_date = datetime.now() # Ensure last_updated is current before saving
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(project.to_dict(), f, indent=2, ensure_ascii=False)

    def load_project(self, project_id: str) -> Optional[QuantumProject]:
        file_path = self._get_project_file_path(project_id)
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return QuantumProject.from_dict(data)
        return None

    def delete_project(self, project_id: str) -> bool:
        file_path = self._get_project_file_path(project_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_all_project_ids(self) -> List[str]:
        return [p.stem for p in self.storage_path.glob("*.json")]

    def load_all_projects(self) -> List[QuantumProject]:
        projects = []
        for project_id in self.list_all_project_ids():
            project = self.load_project(project_id)
            if project:
                projects.append(project)
        return projects
