import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from anthropic import Anthropic
from openai import OpenAI

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.data_models.scenario import Scenario
from synthetic_conversation_generation.llm_queries.character_message_query import CharacterMessageQuery
from synthetic_conversation_generation.llm_queries.conversation_completion_query import ConversationCompletionQuery
from synthetic_conversation_generation.llm_queries.llm_query import (
    ModelProvider,
    OpenAIModelProvider,
    AnthropicModelProvider,
    OllamaModelProvider,
    TransformersModelProvider,
)
from synthetic_conversation_generation.temporal import ConversationTimer

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)


@dataclass
class PhaseSchedule:
    """
    Defines when the conversation shifts relationship phase based on elapsed days.
    Default schedule models a realistic arc over 14 days.
    """
    transitions: List[tuple] = field(default_factory=lambda: [
        (0,  "early_contact"),
        (3,  "escalation"),
        (10, "post_incident"),
        (12, "re_initiation"),
    ])

    def phase_for(self, elapsed_days: float) -> str:
        current_phase = self.transitions[0][1]
        for day_threshold, phase in self.transitions:
            if elapsed_days >= day_threshold:
                current_phase = phase
        return current_phase


class ConversationGenerator:
    """
    Generates a synthetic text message conversation between two characters
    placed in a given scenario.

    Characters and scenario are specified independently — any two CharacterCards
    can be combined with any Scenario. The model is responsible for figuring out
    how these people interact given the situation.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        model_id: str,
        character_a: CharacterCard,
        character_b: CharacterCard,
        scenario: Scenario,
        max_conversation_turns: int = 20,
        phase_schedule: PhaseSchedule | None = None,
        conversation_start_time: datetime | None = None,
        hawkes_seed: int | None = None,
    ):
        self.model_provider = model_provider
        self.model_id = model_id
        self.character_a = character_a   # sends first
        self.character_b = character_b   # replies
        self.scenario = scenario
        self.max_conversation_turns = max_conversation_turns
        self.phase_schedule = phase_schedule or PhaseSchedule()
        self.conversation_start_time = conversation_start_time or datetime(2024, 1, 1, 9, 0)
        self.hawkes_seed = hawkes_seed

    def generate_conversation(self, conversation_id: str) -> Conversation:
        conversation = Conversation(
            id=str(conversation_id),
            user_id=self.character_a.name,
            messages=[]
        )

        timer = ConversationTimer(
            start_time=self.conversation_start_time,
            phase=self.phase_schedule.phase_for(0),
            seed=self.hawkes_seed,
        )

        for i in range(self.max_conversation_turns):
            # Update phase based on elapsed time
            current_phase = self.phase_schedule.phase_for(timer.elapsed_days)
            if current_phase != timer.phase:
                logger.info(f"Turn {i}: phase {timer.phase} → {current_phase} (day {timer.elapsed_days:.1f})")
                timer.set_phase(current_phase)

            # Alternate sender each turn: even = character_a, odd = character_b
            is_sender_a = (i % 2 == 0)
            if is_sender_a:
                sender, receiver = self.character_a, self.character_b
                role = ROLE.user
            else:
                sender, receiver = self.character_b, self.character_a
                role = ROLE.assistant

            next_ts, gap_minutes = timer.next_timestamp()
            logger.info(f"Turn {i} | {sender.name} | phase={timer.phase} | gap={gap_minutes:.1f}min | {next_ts.strftime('%Y-%m-%d %H:%M')}")

            message = CharacterMessageQuery(
                model_provider=self.model_provider,
                model_id=self.model_id,
                conversation=conversation,
                sender=sender,
                receiver=receiver,
                scenario=self.scenario,
                is_sender_character_a=is_sender_a,
                next_timestamp=next_ts,
                gap_minutes=gap_minutes,
            ).query()

            message.role = role
            conversation.messages.append(message)

            # Check for natural end every two turns (one full exchange)
            if i % 2 == 1:
                is_complete = ConversationCompletionQuery(
                    model_provider=self.model_provider,
                    model_id=self.model_id,
                    conversation=conversation,
                    character_a=self.character_a,
                    character_b=self.character_b,
                ).query()
                if is_complete:
                    logger.info(f"Conversation {conversation_id} complete at turn {i}")
                    break

        return conversation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--character-a", type=str,
                        default="data/characters/priya_sharma.yaml",
                        help="YAML file for character_a (sends first)")
    parser.add_argument("--character-b", type=str,
                        default="data/characters/james_whitmore.yaml",
                        help="YAML file for character_b (replies)")
    parser.add_argument("--scenario", type=str,
                        default="data/scenarios/microaggression_sexism.yaml",
                        help="YAML file describing the scenario")
    parser.add_argument("--output-path", type=str,
                        default="data/conversations/microaggression_run.jsonl")
    parser.add_argument("--model-provider", type=str,
                        choices=["openai", "anthropic", "ollama", "transformers"],
                        default="ollama")
    parser.add_argument("--model-id", type=str, default="llama3:latest")
    parser.add_argument("--max-conversation-turns", type=int, default=20)
    parser.add_argument("--hawkes-seed", type=int, default=None)
    args = parser.parse_args()

    if args.model_provider == "openai":
        model_provider = OpenAIModelProvider(OpenAI())
    elif args.model_provider == "anthropic":
        model_provider = AnthropicModelProvider(Anthropic())
    elif args.model_provider == "ollama":
        model_provider = OllamaModelProvider()
    else:
        model_provider = TransformersModelProvider(model_id=args.model_id)

    character_a = CharacterCard.from_yaml(args.character_a)
    character_b = CharacterCard.from_yaml(args.character_b)
    scenario = Scenario.from_yaml(args.scenario)

    logger.info(f"Generating conversation: {character_a.name} ↔ {character_b.name} | scenario: {scenario.title}")

    generator = ConversationGenerator(
        model_provider=model_provider,
        model_id=args.model_id,
        character_a=character_a,
        character_b=character_b,
        scenario=scenario,
        max_conversation_turns=args.max_conversation_turns,
        hawkes_seed=args.hawkes_seed,
    )

    conversation = generator.generate_conversation("001")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        formatted_messages = []
        for msg in conversation.messages:
            name = character_a.name if msg.role == ROLE.user else character_b.name
            formatted_messages.append({
                "speaker": name,
                "role": msg.role.name,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M"),
                "content": msg.content,
            })
        f.write(json.dumps({
            "conversation_id": "001",
            "scenario": scenario.title,
            "characters": [character_a.name, character_b.name],
            "messages": formatted_messages,
        }, indent=2))

    print(f"\nSaved {len(conversation.messages)} messages to {args.output_path}")
