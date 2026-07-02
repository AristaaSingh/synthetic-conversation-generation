from dataclasses import dataclass
from typing import Dict

import yaml


@dataclass
class Scenario:
    """
    Describes the situation two characters are placed in.
    Kept separate from character cards so any two characters can be
    combined with any scenario without rewriting either.
    """
    title: str
    description: str           # what is happening / the setup
    relationship: str          # how character_a and character_b know each other
    vawg_category: str         # e.g. STEREOTYPING-DOMINANCE
    character_a_context: str   # situation-specific framing for character_a
    character_b_context: str   # situation-specific framing for character_b

    @classmethod
    def from_dict(cls, data: Dict) -> "Scenario":
        return cls(
            title=data["title"],
            description=data.get("description", ""),
            relationship=data.get("relationship", ""),
            vawg_category=data.get("vawg_category", ""),
            character_a_context=data.get("character_a_context", ""),
            character_b_context=data.get("character_b_context", ""),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Scenario":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
