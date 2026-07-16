from dataclasses import dataclass, field
from typing import Optional

import yaml

from synthetic_conversation_generation.data_models.microaggression_taxonomy import (
    CATEGORY_KEYS,
    TAXONOMY,
    is_valid,
)


@dataclass
class World:
    """
    The setting a character pair is dropped into.

    `vawg_categories` declares which canonical microaggression categories are in
    scope for this world + character pairing — the *palette* the dialogue-flow
    planner may draw from. It replaces the previous free-text `vawg_category`
    string, which held EXIST Subtask-3 labels (e.g. "STEREOTYPING-DOMINANCE").
    That was a different taxonomy from the project's canonical scheme, and was
    injected as a bare label with no definition — the weak configuration that
    Kumar et al. (cited in Lagos Rojas et al., CHI 2026) show materially reduces
    an LLM's sensitivity. Categories are now validated against
    `microaggression_taxonomy` and always travel with their definitions.
    """
    title: str
    setting: str
    relationship: str
    vawg_categories: list[str]
    character_a_role: str
    character_b_role: str
    character_a_context: Optional[str] = None
    character_b_context: Optional[str] = None

    def __post_init__(self):
        if not self.vawg_categories:
            raise ValueError(
                f"World '{self.title}' declares no vawg_categories. "
                f"Expected one or more of: {CATEGORY_KEYS}"
            )
        unknown = [c for c in self.vawg_categories if not is_valid(c)]
        if unknown:
            raise ValueError(
                f"World '{self.title}' declares unknown vawg_categories: {unknown}. "
                f"Valid categories: {CATEGORY_KEYS}"
            )

    def category_definitions(self) -> str:
        """The in-scope categories rendered with definitions, for prompt injection."""
        lines = []
        for key in self.vawg_categories:
            c = TAXONOMY[key]
            lines.append(f"- {c.key} ({c.label}): {c.definition}")
            lines.append(f"    In a workplace: {c.workplace_form}")
        return "\n".join(lines)

    @classmethod
    def from_yaml(cls, path: str) -> "World":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            title=data["title"],
            setting=data.get("setting", ""),
            relationship=data.get("relationship", ""),
            vawg_categories=data.get("vawg_categories", []),
            character_a_role=data.get("character_a_role", ""),
            character_b_role=data.get("character_b_role", ""),
            character_a_context=data.get("character_a_context"),
            character_b_context=data.get("character_b_context"),
        )
