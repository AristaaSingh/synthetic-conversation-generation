from dataclasses import dataclass
from typing import Optional

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.data_models.world import World
from synthetic_conversation_generation.llm_queries.llm_query import LLMQuery, ModelProvider


@dataclass
class RollingSummary:
    events: str        # what has happened and been decided
    details: str       # specific names, projects, decisions introduced
    open_threads: str  # topics mentioned but not resolved
    dynamic: str       # how the two characters are relating; any VAWG-relevant patterns


class RollingSummaryQuery(LLMQuery):
    """
    Compresses earlier turns into a structured summary that replaces raw
    history in the generation prompt.

    Run every SUMMARY_INTERVAL turns. Covers all messages up to
    (current_turn - RECENT_TURNS_KEPT), leaving the most recent turns
    as raw context in the prompt.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        conversation: Conversation,
        character_a: CharacterCard,
        character_b: CharacterCard,
        world: World,
        summarise_up_to_index: int,
        previous_summary: Optional[RollingSummary] = None,
    ):
        super().__init__(model_provider, model_id)
        self.conversation = conversation
        self.character_a = character_a
        self.character_b = character_b
        self.world = world
        self.summarise_up_to_index = summarise_up_to_index
        self.previous_summary = previous_summary

    def generate_prompt(self):
        messages = self.conversation.messages[:self.summarise_up_to_index]
        history_lines = []
        for msg in messages:
            name = self.character_a.name if msg.role == ROLE.user else self.character_b.name
            history_lines.append(
                f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] {name}: {msg.content}"
            )

        prior = ""
        if self.previous_summary:
            prior = f"""Previous summary (update this, don't just repeat it):
Events: {self.previous_summary.events}
Details introduced: {self.previous_summary.details}
Open threads: {self.previous_summary.open_threads}
Relationship dynamic: {self.previous_summary.dynamic}

"""

        return f"""Summarise this text message conversation between {self.character_a.name} and {self.character_b.name}.

{prior}New messages to incorporate:
{chr(10).join(history_lines)}

Produce a structured summary with four fields:

events — what has happened and been decided (include specific outcomes, not just topics)

details — specific names, projects, files, tools, numbers, decisions mentioned that should be remembered

open_threads — ONLY things where no conclusion has been reached and active follow-up is genuinely needed. Do NOT include plans that have already been mutually agreed and scheduled for a specific future time — those are decided, not pending. Remove any thread from this list once both people have acknowledged it and committed to a time or action. If a thread has appeared in previous open_threads but both parties have now agreed on it, drop it entirely.

dynamic — how {self.character_a.name} and {self.character_b.name} are actually relating to each other beneath the surface. Note any patterns in how {self.character_b.name} treats {self.character_a.name}, any moments of friction or power imbalance, anything left unsaid. Be specific — do not just say "they are collaborating well."
"""

    def response_schema(self):
        return {
            "type": "object",
            "properties": {
                "events":        {"type": "string"},
                "details":       {"type": "string"},
                "open_threads":  {"type": "string"},
                "dynamic":       {"type": "string"},
            },
            "required": ["events", "details", "open_threads", "dynamic"],
            "additionalProperties": False,
        }

    def parse_response(self, json_response) -> RollingSummary:
        return RollingSummary(
            events=json_response.get("events", ""),
            details=json_response.get("details", ""),
            open_threads=json_response.get("open_threads", ""),
            dynamic=json_response.get("dynamic", ""),
        )
