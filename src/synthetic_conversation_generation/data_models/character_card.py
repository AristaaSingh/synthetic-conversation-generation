from typing import Dict

from dataclasses import dataclass

import yaml


@dataclass
class CharacterCard:
    name: str
    backstory: str
    description: str
    personality: str
    summary: str

    @classmethod
    def from_dict(cls, data: Dict) -> "CharacterCard":
        return cls(
            name=data["name"],
            backstory=data.get("backstory", ""),
            description=data.get("description", ""),
            personality=data.get("personality", ""),
            summary=data.get("summary", ""),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "CharacterCard":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
