from dataclasses import dataclass
from typing import Optional

import yaml


@dataclass
class World:
    title: str
    setting: str
    relationship: str
    vawg_category: str
    character_a_role: str
    character_b_role: str
    character_a_context: Optional[str] = None
    character_b_context: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "World":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            title=data["title"],
            setting=data.get("setting", ""),
            relationship=data.get("relationship", ""),
            vawg_category=data.get("vawg_category", ""),
            character_a_role=data.get("character_a_role", ""),
            character_b_role=data.get("character_b_role", ""),
            character_a_context=data.get("character_a_context"),
            character_b_context=data.get("character_b_context"),
        )
