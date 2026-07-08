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
from synthetic_conversation_generation.data_models.world import World
from synthetic_conversation_generation.llm_queries.character_message_query import CharacterMessageQuery
from synthetic_conversation_generation.llm_queries.conversation_completion_query import ConversationCompletionQuery
from synthetic_conversation_generation.llm_queries.llm_query import (
    ModelProvider,
    OpenAIModelProvider,
    AnthropicModelProvider,
    OllamaModelProvider,
    TransformersModelProvider,
)
from synthetic_conversation_generation.llm_queries.rolling_summary_query import RollingSummaryQuery, RollingSummary
from synthetic_conversation_generation.llm_queries.state_assessment_query import StateAssessmentQuery
from synthetic_conversation_generation.temporal import ConversationTimer

# Summarise every N turns, keeping the most recent M turns as raw context.
_SUMMARY_INTERVAL = 10
_RECENT_TURNS_KEPT = 10

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
    world: World,
    conversation_id: str,
    max_turns: int = 60,
    max_sessions: int = 6,
    conversation_start_time: datetime | None = None,
    hawkes_seed: int | None = None,
) -> Conversation:
    """
    Generate a single synthetic conversation spanning multiple sessions.

    character_a sends first. Turns alternate. After every full exchange
    (one message from each character), a StateAssessmentQuery runs to:
      1. Update the narrative state summary (context for the next turn)
      2. Determine the current Hawkes phase (event-driven transition)

    When the ConversationCompletionQuery detects a natural sign-off,
    that is treated as a SESSION boundary rather than the end of the
    conversation. A new session starts after a realistic gap (hours to
    days). The conversation only ends when max_sessions is reached or
    max_turns is exhausted.
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
    session_count = 0
    rolling_summary: RollingSummary | None = None

    for i in range(max_turns):
        is_sender_a = (i % 2 == 0)
        sender = character_a if is_sender_a else character_b
        receiver = character_b if is_sender_a else character_a
        role = ROLE.user if is_sender_a else ROLE.assistant

        next_ts, gap_minutes = timer.next_timestamp()
        logger.info(
            f"Turn {i} | session {session_count + 1}/{max_sessions} | {sender.name} | "
            f"phase={state.phase} | tension={state.tension_level}/5 | "
            f"gap={gap_minutes:.1f}min | {next_ts.strftime('%Y-%m-%d %H:%M')}"
        )

        message = CharacterMessageQuery(
            model_provider=model_provider,
            model_id=model_id,
            conversation=conversation,
            sender=sender,
            receiver=receiver,
            world=world,
            is_sender_character_a=is_sender_a,
            next_timestamp=next_ts,
            gap_minutes=gap_minutes,
            state_summary=state.summary,
            rolling_summary=rolling_summary,
        ).query()

        message.role = role
        conversation.messages.append(message)

        # After each full exchange: update rolling summary, assess state, check session end
        if i % 2 == 1:
            # Rolling summary: compress older turns every SUMMARY_INTERVAL turns
            total_turns = len(conversation.messages)
            if total_turns >= _SUMMARY_INTERVAL and total_turns % _SUMMARY_INTERVAL == 0:
                summarise_up_to = total_turns - _RECENT_TURNS_KEPT
                if summarise_up_to > 0:
                    logger.info(f"Running rolling summary over turns 0–{summarise_up_to}")
                    rolling_summary = RollingSummaryQuery(
                        model_provider=model_provider,
                        model_id=model_id,
                        conversation=conversation,
                        character_a=character_a,
                        character_b=character_b,
                        world=world,
                        summarise_up_to_index=summarise_up_to,
                        previous_summary=rolling_summary,
                    ).query()

            new_state = StateAssessmentQuery(
                model_provider=model_provider,
                model_id=model_id,
                conversation=conversation,
                character_a=character_a,
                character_b=character_b,
                world=world,
                previous_state=state,
            ).query()

            if new_state.phase != state.phase:
                logger.info(f"Phase transition: {state.phase} → {new_state.phase} (tension {new_state.tension_level}/5)")
                timer.set_phase(new_state.phase)

            state = new_state

            session_ended = ConversationCompletionQuery(
                model_provider=model_provider,
                model_id=model_id,
                conversation=conversation,
                character_a=character_a,
                character_b=character_b,
            ).query()

            if session_ended:
                session_count += 1
                logger.info(f"Session {session_count} ended at turn {i} ({next_ts.strftime('%Y-%m-%d %H:%M')})")

                if session_count >= max_sessions:
                    logger.info(f"Reached max_sessions ({max_sessions}). Conversation complete.")
                    break

                # Session boundary: jump forward by hours or days before the next session.
                # The timer's Hawkes state is reset with a large forced gap so the next
                # exchange starts fresh rather than continuing the previous burst.
                timer.force_gap_hours(between=4, spread=20)
                logger.info(f"Starting session {session_count + 1} — next message around {timer.current_time.strftime('%Y-%m-%d %H:%M')}")

    return conversation, state, rolling_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic VAWG conversation pipeline")
    parser.add_argument("--character-a", type=str,
                        default="data/characters/victims/priya_sharma.yaml",
                        help="YAML file for character_a — always the victim")
    parser.add_argument("--character-b", type=str,
                        default="data/characters/perpetrators/james_whitmore.yaml",
                        help="YAML file for character_b — always the perpetrator")
    parser.add_argument("--world", type=str,
                        default="data/worlds/uk_tech_company.yaml",
                        help="YAML file describing the world")
    parser.add_argument("--output-path", type=str,
                        default="data/conversations/output.json")
    parser.add_argument("--model-provider", type=str,
                        choices=["openai", "anthropic", "ollama", "transformers"],
                        default="ollama")
    parser.add_argument("--model-id", type=str, default="llama3:latest")
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-sessions", type=int, default=6)
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
    world = World.from_yaml(args.world)

    logger.info(f"Starting pipeline: {character_a.name} ↔ {character_b.name} | {world.title}")

    conversation, final_state, rolling_summary = run_pipeline(
        model_provider=model_provider,
        model_id=args.model_id,
        character_a=character_a,
        character_b=character_b,
        world=world,
        conversation_id=args.conversation_id,
        max_turns=args.max_turns,
        max_sessions=args.max_sessions,
        hawkes_seed=args.hawkes_seed,
    )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "conversation_id": args.conversation_id,
        "world": world.title,
        "characters": [character_a.name, character_b.name],
        "rolling_summary": {
            "events": rolling_summary.events,
            "details": rolling_summary.details,
            "open_threads": rolling_summary.open_threads,
            "dynamic": rolling_summary.dynamic,
        } if rolling_summary else None,
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
