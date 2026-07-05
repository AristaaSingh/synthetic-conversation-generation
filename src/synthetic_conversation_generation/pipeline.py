"""
pipeline.py — main entry point for synthetic VAWG conversation generation.

Wires together:
  - CharacterCard + Scenario (inputs)
  - ConversationTimer (Hawkes process inter-message timing)
  - CharacterMessageQuery (generates each message)
  - StateAssessmentQuery (assesses relational state after each exchange)
  - ConversationCompletionQuery (decides when the conversation has ended)

Phase transitions in the Hawkes process are event-driven: the StateAssessmentQuery
determines which phase the conversation is in based on what has actually happened,
and the timer updates accordingly. There are no hardcoded day thresholds.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.data_models.conversation_state import ConversationState
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
from synthetic_conversation_generation.llm_queries.state_assessment_query import StateAssessmentQuery
from synthetic_conversation_generation.temporal import ConversationTimer

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)


def run_pipeline(
    model_provider: ModelProvider,
    model_id: str,
    character_a: CharacterCard,
    character_b: CharacterCard,
    scenario: Scenario,
    conversation_id: str,
    max_turns: int = 40,
    conversation_start_time: datetime | None = None,
    hawkes_seed: int | None = None,
) -> Conversation:
    """
    Generate a single synthetic conversation.

    character_a sends first. Turns alternate. After every full exchange
    (one message from each character), a StateAssessmentQuery runs to:
      1. Update the narrative state summary (context for the next turn)
      2. Determine the current Hawkes phase (event-driven transition)
    """
    conversation = Conversation(
        id=conversation_id,
        user_id=character_a.name,
        messages=[],
    )

    start_time = conversation_start_time or datetime(2024, 1, 1, 9, 0)
    initial_state = ConversationState(
        phase="early_contact",
        summary="The conversation has just started. No significant events have occurred yet.",
        tension_level=1,
        incident_occurred=False,
    )

    timer = ConversationTimer(
        start_time=start_time,
        phase=initial_state.phase,
        seed=hawkes_seed,
    )

    state = initial_state

    for i in range(max_turns):
        is_sender_a = (i % 2 == 0)
        sender = character_a if is_sender_a else character_b
        receiver = character_b if is_sender_a else character_a
        role = ROLE.user if is_sender_a else ROLE.assistant

        next_ts, gap_minutes = timer.next_timestamp()
        logger.info(
            f"Turn {i} | {sender.name} | phase={state.phase} | "
            f"tension={state.tension_level}/5 | gap={gap_minutes:.1f}min | "
            f"{next_ts.strftime('%Y-%m-%d %H:%M')}"
        )

        message = CharacterMessageQuery(
            model_provider=model_provider,
            model_id=model_id,
            conversation=conversation,
            sender=sender,
            receiver=receiver,
            scenario=scenario,
            is_sender_character_a=is_sender_a,
            next_timestamp=next_ts,
            gap_minutes=gap_minutes,
            state_summary=state.summary,
        ).query()

        message.role = role
        conversation.messages.append(message)

        # After each full exchange: assess state, update phase, check completion
        if i % 2 == 1:
            new_state = StateAssessmentQuery(
                model_provider=model_provider,
                model_id=model_id,
                conversation=conversation,
                character_a=character_a,
                character_b=character_b,
                previous_state=state,
            ).query()

            if new_state.phase != state.phase:
                logger.info(f"Phase transition: {state.phase} → {new_state.phase} (tension {new_state.tension_level}/5)")
                timer.set_phase(new_state.phase)

            state = new_state

            is_complete = ConversationCompletionQuery(
                model_provider=model_provider,
                model_id=model_id,
                conversation=conversation,
                character_a=character_a,
                character_b=character_b,
            ).query()

            if is_complete:
                logger.info(f"Conversation {conversation_id} ended naturally at turn {i}")
                break

    return conversation, state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic VAWG conversation pipeline")
    parser.add_argument("--character-a", type=str,
                        default="data/characters/priya_sharma.yaml",
                        help="YAML file for character_a (sends first)")
    parser.add_argument("--character-b", type=str,
                        default="data/characters/james_whitmore.yaml",
                        help="YAML file for character_b")
    parser.add_argument("--scenario", type=str,
                        default="data/scenarios/microaggression_sexism.yaml",
                        help="YAML file describing the scenario")
    parser.add_argument("--output-path", type=str,
                        default="data/conversations/output.json")
    parser.add_argument("--model-provider", type=str,
                        choices=["openai", "anthropic", "ollama", "transformers"],
                        default="ollama")
    parser.add_argument("--model-id", type=str, default="llama3:latest")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--hawkes-seed", type=int, default=None)
    parser.add_argument("--conversation-id", type=str, default="001")
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

    logger.info(f"Starting pipeline: {character_a.name} ↔ {character_b.name} | {scenario.title}")

    conversation, final_state = run_pipeline(
        model_provider=model_provider,
        model_id=args.model_id,
        character_a=character_a,
        character_b=character_b,
        scenario=scenario,
        conversation_id=args.conversation_id,
        max_turns=args.max_turns,
        hawkes_seed=args.hawkes_seed,
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "conversation_id": args.conversation_id,
        "scenario": scenario.title,
        "characters": [character_a.name, character_b.name],
        "final_state": {
            "phase": final_state.phase,
            "summary": final_state.summary,
            "tension_level": final_state.tension_level,
            "incident_occurred": final_state.incident_occurred,
        },
        "messages": [
            {
                "speaker": character_a.name if msg.role == ROLE.user else character_b.name,
                "role": msg.role.name,
                "timestamp": msg.timestamp.strftime("%Y-%m-%d %H:%M"),
                "content": msg.content,
            }
            for msg in conversation.messages
        ],
    }

    with open(args.output_path, "w") as f:
        f.write(json.dumps(output, indent=2))

    print(f"Saved {len(conversation.messages)} messages to {args.output_path}")
    print(f"Final state: {final_state.phase} | tension {final_state.tension_level}/5 | incident: {final_state.incident_occurred}")
