from typing import Dict, List, Optional

from dataclasses import dataclass


@dataclass
class CharacterCard:
    name: str
    role: str  # e.g., "victim" or "perpetrator"
    backstory: str
    description: str
    personality: str
    scenario: str
    summary: str

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Create a CharacterCard instance from a dictionary.
        
        Args:
            data: Dictionary containing character data
            
        Returns:
            CharacterCard instance
        """
        return cls(
            name=data['name'],
            role=data.get('role', 'unspecified'),
            backstory=data.get('backstory', ''),
            description=data.get('description', ''),
            personality=data.get('personality', ''),
            scenario=data.get('scenario', ''),
            summary=data.get('summary', '')
        )
