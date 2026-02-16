import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

@dataclass
class QuantumProject:
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Quantum Project"
    status: str = "Planning" # E.g., Planning, Active Development, Testing, Operational, Archived, ZEN_Incubation
    creator: str = "Greg"
    objective: str = ""
    
    primary_frequency_hz: Optional[float] = None
    target_coherence: Optional[Any] = None # Can be float (e.g., 0.95) or str (e.g., "φ", "φ^φ")
    associated_harmonics: List[str] = field(default_factory=list) # e.g., ["φ^2", "7.83 Hz"]
    
    quantum_components_used: List[str] = field(default_factory=list) # e.g., ["QuantumCore", "ZenPointField"]
    wizdome_era: Optional[str] = None # E.g., BCE, CE, CASCADE_Era
    being_phi_state: Optional[str] = None # E.g., Recognition, Redirection, Incubation, Integration
    
    start_date: datetime = field(default_factory=datetime.now)
    last_updated_date: datetime = field(default_factory=datetime.now)
    target_completion_date: Optional[datetime] = None
    
    associated_files_paths: List[str] = field(default_factory=list)
    experiment_ids: List[str] = field(default_factory=list)
    
    know_link_references: List[str] = field(default_factory=list) # Links to WIZDOME docs or principles
    notes: str = "" # Free-form text, can be extensive

    def __post_init__(self):
        # Ensure datetime fields are actual datetime objects if loaded from string representations
        if isinstance(self.start_date, str):
            try:
                self.start_date = datetime.fromisoformat(self.start_date)
            except ValueError:
                # Handle cases where it might not be a perfect ISO format, or log error
                pass # Or set to now(), or raise error, depending on desired strictness
        if isinstance(self.last_updated_date, str):
            try:
                self.last_updated_date = datetime.fromisoformat(self.last_updated_date)
            except ValueError:
                pass
        if self.target_completion_date and isinstance(self.target_completion_date, str):
            try:
                self.target_completion_date = datetime.fromisoformat(self.target_completion_date)
            except ValueError:
                self.target_completion_date = None # Or handle error

    def to_dict(self) -> Dict[str, Any]:
        # Convert to dict, ensuring datetime objects are serialized to ISO format strings
        data = self.__dict__.copy()
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumProject':
        # Dataclasses can often be instantiated directly from dicts if types match
        # but __post_init__ will handle datetime string conversions.
        return cls(**data)
