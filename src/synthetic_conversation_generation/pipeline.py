"""
pipeline.py — main entry point for synthetic VAWG conversation generation.

Wires together:
  - CharacterCard + Scenario (inputs)
  - ConversationTimer (Hawkes process inter-message timing)
  - DialogueFlowQuery (pre-plans each session's beats)
  - CharacterMessageQuery (generates each message)
  - PersonaConsistencyQuery (rejects off-character messages)
  - StateAssessmentQuery (relational state, and whether the session has ended)

Phase transitions in the Hawkes process are event-driven: the StateAssessmentQuery
determines which phase the conversation is in based on what has actually happened,
and the timer updates accordingly. There are no hardcoded day thresholds.
"""
import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from openai import OpenAI

from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.commitment_cache import CommitmentCache
from synthetic_conversation_generation.data_models.conversation import Conversation, ROLE
from synthetic_conversation_generation.data_models.conversation_state import ConversationState
from synthetic_conversation_generation.data_models.world import World
from synthetic_conversation_generation.data_models.dialogue_flow import DialogueFlow
from synthetic_conversation_generation.llm_queries.character_message_query import CharacterMessageQuery
from synthetic_conversation_generation.llm_queries.commitment_extraction_query import CommitmentExtractionQuery
from synthetic_conversation_generation.llm_queries.dialogue_flow_query import DialogueFlowQuery
from synthetic_conversation_generation.llm_queries.llm_query import (
    ModelProvider,
    OpenAIModelProvider,
    AnthropicModelProvider,
    OllamaModelProvider,
    TransformersModelProvider,
)
from synthetic_conversation_generation.llm_queries.persona_consistency_query import PersonaConsistencyQuery
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


def build_output(
    conversation: Conversation,
    state: ConversationState,
    rolling_summary: "RollingSummary | None",
    all_dialogue_flows: list[DialogueFlow],
    commitment_cache: CommitmentCache,
    character_a: CharacterCard,
    character_b: CharacterCard,
    world: World,
    conversation_id: str,
    complete: bool,
) -> dict:
    """
    Serialise the full run state to a JSON-ready dict.

    Used both for periodic checkpoints during generation and for the final write,
    so a checkpoint file has exactly the same shape as a completed one — only the
    `complete` flag differs.
    """
    return {
        "conversation_id": conversation_id,
        "world": world.title,
        "characters": [character_a.name, character_b.name],
        # False if this file is a mid-run checkpoint (e.g. the job was killed by a
        # SLURM timeout). Consumers must check this before treating a run as finished.
        "complete": complete,
        "turns_generated": len(conversation.messages),
        "vawg_categories": list(world.vawg_categories),
        "commitment_cache": commitment_cache.to_dict_list(),
        "dialogue_flows": [
            {
                "session_number": flow.session_number,
                "planned_turns": flow.total_turns(),
                "beats": [
                    {
                        "topic": b.topic,
                        "category": b.category,
                        "severity": b.severity,
                        "exchanges": b.exchanges,
                        "description": b.description,
                    }
                    for b in flow.beats
                ],
            }
            for flow in all_dialogue_flows
        ],
        "rolling_summary": {
            "events": rolling_summary.events,
            "details": rolling_summary.details,
            "open_threads": rolling_summary.open_threads,
            "dynamic": rolling_summary.dynamic,
        } if rolling_summary else None,
        "final_state": {
            "phase": state.phase,
            "summary": state.summary,
            "tension_level": state.tension_level,
            "incident_occurred": state.incident_occurred,
            "detected_categories": state.detected_categories,
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


def write_output(path: Path, data: dict) -> None:
    """
    Write the output JSON atomically.

    A plain `open(path, "w")` followed by `json.dump` is not safe here: a run can be
    killed mid-write by a SLURM timeout, which would leave a truncated, unparseable
    file — and, for checkpoints, would destroy the previously good checkpoint it was
    overwriting. Writing to a temporary file in the same directory and then calling
    os.replace() makes the swap atomic on POSIX, so the destination always contains
    either the previous complete checkpoint or the new one, never a half-written mix.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(data, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_pipeline(
    model_provider: ModelProvider,
    model_id: str,
    character_a: CharacterCard,
    character_b: CharacterCard,
    world: World,
    conversation_id: str,
    target_days: float = 14.0,
    max_turns: int = 300,
    conversation_start_time: datetime | None = None,
    hawkes_seed: int | None = None,
    output_path: Path | None = None,
    exchange_budget: int = 5,
) -> Conversation:
    """
    Generate a single synthetic conversation spanning multiple sessions.

    character_a sends first. Turns alternate. After every full exchange
    (one message from each character), a StateAssessmentQuery runs to:
      1. Update the narrative state summary (context for the next turn)
      2. Determine the current Hawkes phase (event-driven transition)

    A session ends when EITHER the assessor judges the conversation has reached a
    natural stopping point (the primary, emergent trigger) OR the session's beat
    plan is spent (a backstop). That is treated as a SESSION boundary rather than
    the end of the conversation: the clock jumps forward hours-to-a-day and a fresh
    beat plan is drawn up, informed by the tension and summary accumulated so far.
    Sessions are unbounded — there is no session ceiling.

    Termination
    -----------
    The conversation ends when the SIMULATED CLOCK passes `target_days`.

    This replaces the previous `max_turns` / `max_sessions` ceilings, which were
    proxies for a goal they only loosely tracked. The project's aim is an arc
    spanning ~2 weeks, which is a statement about *duration*, not message count:
    100 turns spanned 6d16h while the relationship stayed in escalation, but the
    same 100 turns in post_incident (44h session gaps) would have spanned ~40 days.
    Capping messages therefore cut conversations off at a number unrelated to the
    goal, and forced every conversation to be the same length regardless of what
    happened in it — itself a visible artefact in a corpus.

    Terminating on duration makes turn count *emergent and variable*, which is the
    realistic behaviour: a tense fortnight generates more messages than a withdrawn
    one.

    `max_turns` is retained only as a CIRCUIT BREAKER against a runaway run eating
    the SLURM wall clock. It is set generously so that it does not bind in normal
    operation; if it fires, that is a signal worth investigating, not a normal exit.
    """
    conversation = Conversation(
        id=conversation_id,
        user_id=character_a.name,
        messages=[],
    )

    # Fixed start time across all conversations — a deliberate simplification (see
    # project_record.md 22.4). Monday 09:00 is chosen because the model *sees* this
    # date on every history line, so it must be a plausible working morning:
    # 1 January would be New Year's Day, and a weekend start would make the
    # workplace content incoherent.
    start_time = conversation_start_time or datetime(2026, 1, 5, 9, 0)
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

    # Per-session planning budget, in exchanges (1 exchange = 2 turns). Now a direct
    # parameter rather than being derived from max_sessions: with sessions unbounded
    # there is no session count to divide by, and the old derivation coupled two
    # unrelated things (raising max_sessions silently shortened every session).
    state = initial_state
    session_count = 0
    rolling_summary: RollingSummary | None = None
    all_dialogue_flows: list[DialogueFlow] = []
    # Application-layer commitment cache (LMCache-inspired, Liu et al., 2025):
    # persists explicit instructions across summary compression boundaries.
    commitment_cache = CommitmentCache()


    def checkpoint(complete: bool = False) -> None:
        """Persist current progress so a timeout or crash does not lose the run."""
        if output_path is None:
            return
        write_output(
            output_path,
            build_output(
                conversation=conversation,
                state=state,
                rolling_summary=rolling_summary,
                all_dialogue_flows=all_dialogue_flows,
                commitment_cache=commitment_cache,
                character_a=character_a,
                character_b=character_b,
                world=world,
                conversation_id=conversation_id,
                complete=complete,
            ),
        )

    # Dialogue flow pre-planning (SynDG, Bao et al., ACL 2023):
    # the session's topic arc is determined before any messages are generated.
    # Beat severity tiers follow the STOP offensive progression scale
    # (Morabito et al., EMNLP 2024): 1 = neutral → 5 = acute.
    session_count_for_flow = 1
    dialogue_flow: DialogueFlow = DialogueFlowQuery(
        model_provider=model_provider,
        model_id=model_id,
        character_a=character_a,
        character_b=character_b,
        world=world,
        session_number=session_count_for_flow,
        previous_state=state,
        rolling_summary=rolling_summary,
        exchange_budget=exchange_budget,
    ).query()
    all_dialogue_flows.append(dialogue_flow)
    logger.info(
        f"Session {session_count_for_flow} flow planned: "
        + " | ".join(f"[{b.severity}/{b.category or 'none'}] {b.topic}" for b in dialogue_flow.beats)
    )

    # `max_turns` is a circuit breaker, not the termination condition — the loop
    # exits on simulated duration (checked after each exchange, below).
    for i in range(max_turns):
        is_sender_a = (i % 2 == 0)
        sender = character_a if is_sender_a else character_b
        receiver = character_b if is_sender_a else character_a
        role = ROLE.user if is_sender_a else ROLE.assistant

        next_ts, gap_minutes = timer.next_timestamp()
        # None once the plan is spent — the generator then winds down on its own
        # rather than being handed the same final beat over and over.
        current_beat = dialogue_flow.current_beat
        beat_desc = (
            f"beat={dialogue_flow._current_index + 1}/{len(dialogue_flow.beats)} "
            f"(sev={current_beat.severity}, cat={current_beat.category or 'none'})"
            if current_beat else "beat=exhausted"
        )
        logger.info(
            f"Turn {i} | session {session_count + 1} | day {timer.elapsed_days:.1f}/{target_days:g} | "
            f"{sender.name} | "
            f"phase={state.phase} | tension={state.tension_level}/5 | {beat_desc} | "
            f"gap={gap_minutes:.1f}min | {next_ts.strftime('%Y-%m-%d %H:%M')}"
        )

        # Persona consistency filter (PSYDIAL, Han et al., LREC-COLING 2024):
        # generate a candidate message and evaluate it against the character's
        # personality; retry up to _MAX_RETRIES times on rejection.
        _MAX_RETRIES = 3
        accepted_message = None
        for attempt in range(_MAX_RETRIES):
            candidate = CharacterMessageQuery(
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
                current_beat=current_beat,
                commitment_cache=commitment_cache,
            ).query()

            consistency = PersonaConsistencyQuery(
                model_provider=model_provider,
                model_id=model_id,
                character=sender,
                conversation=conversation,
                candidate_message=candidate.content,
                is_character_a=is_sender_a,
                other_character_name=receiver.name,
            ).query()

            if consistency.is_consistent:
                accepted_message = candidate
                if attempt > 0:
                    logger.info(
                        f"Turn {i}: message accepted on attempt {attempt + 1} "
                        f"({sender.name})"
                    )
                break
            else:
                logger.info(
                    f"Turn {i}: persona filter rejected attempt {attempt + 1} "
                    f"for {sender.name} — {consistency.reason}"
                )

        # If all retries failed, accept the last candidate rather than dropping the turn.
        if accepted_message is None:
            logger.warning(
                f"Turn {i}: all {_MAX_RETRIES} attempts rejected for {sender.name}; "
                f"accepting last candidate."
            )
            accepted_message = candidate

        accepted_message.role = role
        conversation.messages.append(accepted_message)

        # Spend one turn on the current beat. The flow advances itself once the beat
        # has had the number of exchanges it asked for — beats are variable length,
        # so this is no longer a fixed every-2-turns rule.
        was_exhausted = dialogue_flow.is_exhausted()
        dialogue_flow.record_turn()
        if not was_exhausted:
            nxt = dialogue_flow.current_beat
            if nxt is None:
                logger.info("Beat plan exhausted — generator will wind down unprompted")
            elif nxt is not current_beat:
                logger.info(
                    f"Beat advanced → {dialogue_flow._current_index + 1}/{len(dialogue_flow.beats)}: "
                    f"[sev={nxt.severity}, cat={nxt.category or 'none'}, "
                    f"{nxt.exchanges}ex] {nxt.topic}"
                )

        # After each full exchange: update rolling summary, assess state, check session end
        if i % 2 == 1:
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

            # Commitment cache population (LMCache-inspired, Liu et al., 2025):
            # scan the last exchange for explicit instructions and persist them
            # so they survive rolling summary compression.
            extraction = CommitmentExtractionQuery(
                model_provider=model_provider,
                model_id=model_id,
                conversation=conversation,
                character_a=character_a,
                character_b=character_b,
                turn_index=len(conversation.messages),
            ).query()
            for entry in extraction.entries:
                commitment_cache.add(entry)
                logger.info(
                    f"Commitment cached: {entry.speaker} → {entry.recipient}: \"{entry.text}\""
                )
            commitment_cache.evict_stale(len(conversation.messages))

            # Persist progress after every exchange. Cheap relative to the 3+ LLM
            # calls that just ran, and means a SLURM timeout costs at most one
            # exchange rather than the entire run.
            checkpoint()

            # Session end. Two independent triggers:
            #   1. The assessor judged the conversation reached a natural stopping
            #      point. This is the primary, emergent mechanism — an ending is a
            #      response to what happened, so it cannot be planned in advance.
            #      It replaces the separate ConversationCompletionQuery, which cost
            #      ~30 LLM calls per run, saw only the last 6 messages, and rarely
            #      fired. The assessor reads the whole conversation and already knows
            #      the tension and phase, so it judges this at zero marginal cost.
            #   2. The beat plan is spent. A backstop: without it the conversation
            #      runs on unplanned to max_turns (previously 48 of 60 turns).
            beats_spent = dialogue_flow.is_exhausted()
            session_ended = state.session_ended or beats_spent

            if session_ended:
                session_count += 1
                reason = "natural stopping point" if state.session_ended else "beat plan spent"
                logger.info(
                    f"Session {session_count} ended at turn {i} ({reason}) "
                    f"({next_ts.strftime('%Y-%m-%d %H:%M')})"
                )

                # Terminate on SIMULATED DURATION. Checked at a session boundary
                # rather than mid-session so the conversation never stops halfway
                # through an exchange. The gap that has just been applied is
                # included, so a boundary that carries the clock past the target
                # ends the run here.
                if timer.elapsed_days >= target_days:
                    logger.info(
                        f"Reached target span ({timer.elapsed_days:.1f}/{target_days:g} days) "
                        f"after {session_count} sessions, {len(conversation.messages)} turns. "
                        f"Conversation complete."
                    )
                    break

                # Gap length now depends on the phase the session ended in: a
                # session that ended in post_incident resumes days later, not the
                # same evening. Session boundaries are the dominant contributor to
                # the conversation's overall span (§22), so this is the main lever
                # on arc length as well as a realism fix.
                gap_h = timer.force_gap_hours()
                logger.info(
                    f"Starting session {session_count + 1} — {gap_h:.1f}h gap "
                    f"(phase={state.phase}) — next message around "
                    f"{timer.current_time.strftime('%Y-%m-%d %H:%M')}"
                )

                # Plan the next session's beat sequence (SynDG: new flow per session).
                session_count_for_flow += 1
                dialogue_flow = DialogueFlowQuery(
                    model_provider=model_provider,
                    model_id=model_id,
                    character_a=character_a,
                    character_b=character_b,
                    world=world,
                    session_number=session_count_for_flow,
                    previous_state=state,
                    rolling_summary=rolling_summary,
                    exchange_budget=exchange_budget,
                ).query()
                all_dialogue_flows.append(dialogue_flow)
                logger.info(
                    f"Session {session_count_for_flow} flow planned: "
                    + " | ".join(f"[{b.severity}/{b.category or 'none'}] {b.topic}" for b in dialogue_flow.beats)
                )

    checkpoint(complete=True)
    return conversation, state, rolling_summary, all_dialogue_flows, commitment_cache


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
    parser.add_argument("--target-days", type=float, default=14.0,
                        help="Simulated days to generate. This is the termination "
                             "condition — the conversation ends when the clock passes it.")
    parser.add_argument("--max-turns", type=int, default=300,
                        help="Circuit breaker only. Set generously; if it fires, the run "
                             "hit an unexpected condition rather than finishing normally.")
    parser.add_argument("--exchange-budget", type=int, default=5,
                        help="Exchanges the planner budgets per session (1 exchange = 2 turns).")
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

    # run_pipeline checkpoints to output_path after every exchange and writes a
    # final complete=True version on exit, so no separate write is needed here.
    conversation, final_state, rolling_summary, all_dialogue_flows, commitment_cache = run_pipeline(
        model_provider=model_provider,
        model_id=args.model_id,
        character_a=character_a,
        character_b=character_b,
        world=world,
        conversation_id=args.conversation_id,
        target_days=args.target_days,
        max_turns=args.max_turns,
        exchange_budget=args.exchange_budget,
        hawkes_seed=args.hawkes_seed,
        output_path=Path(args.output_path),
    )

    print(f"Saved {len(conversation.messages)} messages to {args.output_path}")
    print(f"Final state: {final_state.phase} | tension {final_state.tension_level}/5 | incident: {final_state.incident_occurred}")
