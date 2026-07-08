from dataclasses import dataclass
from typing import Dict

import yaml


@dataclass
class CharacterCard:
    name: str
    physical_description: str
    personality: str

    @classmethod
    def from_dict(cls, data: Dict) -> "CharacterCard":
        return cls(
            name=data["name"],
            physical_description=data.get("physical_description", ""),
            personality=data.get("personality", ""),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "CharacterCard":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
