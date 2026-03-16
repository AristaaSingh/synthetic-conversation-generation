import argparse
import logging
from pathlib import Path
from typing import List, Optional
import yaml

from anthropic import Anthropic
from openai import OpenAI

from synthetic_conversation_generation.llm_queries.llm_query import (
    ModelProvider,
    OpenAIModelProvider,
    AnthropicModelProvider,
    OllamaModelProvider,
    TransformersModelProvider,
)

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.character_card import CharacterCard
from synthetic_conversation_generation.data_models.conversation_characters import ConversationCharacters

from synthetic_conversation_generation.llm_queries.user_persona_query import UserPersonaQuery

# Configure root logger to WARNING to silence third-party libraries
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set loggers within this application to INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('anthropic').setLevel(logging.WARNING)


class PersonaGenerator:
    
    def __init__(self, model_provider: ModelProvider, model_id: str, assistant: Assistant, previous_personas: List[CharacterCard], target_role: str | None = None, scenario_theme: str = "workplace sexism"):
        self.model_provider = model_provider
        self.model_id = model_id
        self.assistant = assistant
        self.previous_personas = previous_personas
        self.target_role = target_role
        self.scenario_theme = scenario_theme

    def generate_persona(self) -> CharacterCard:
        user_persona_generator = UserPersonaQuery(
            self.model_provider,
            self.model_id,
            self.assistant,
            self.previous_personas,
            target_role=self.target_role,
            scenario_theme=self.scenario_theme,
        )
        return user_persona_generator.query(max_retries=3, timeout=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant-path", type=str, required=False, default="data/assistants/vawg_dialog_gen.yaml", help="Path to YAML file containing assistant definition")
    parser.add_argument("--output-path", type=str, required=False, default="data/conversation_characters/vawg_personas.yaml", help="Path to save the generated personas (YAML format)")
    parser.add_argument("--model-provider", type=str, choices=["openai", "anthropic", "ollama", "transformers"], default="openai", help="LLM provider to use for generating personas")
    parser.add_argument("--model-id", type=str, default="o3", help="Model ID for persona generation (or local model name for ollama/transformers)")
    parser.add_argument("--transformers-device", type=str, default="auto", help="Device map for transformers pipeline (only when --model-provider transformers)")
    parser.add_argument("--previous-personas-path", type=str, help="Path to YAML file containing previous personas to avoid duplication")
    parser.add_argument("--scenario", type=str, default="workplace sexism", help="Scenario theme to focus personas on (e.g., workplace sexism, microaggressions, online harassment)")
    args = parser.parse_args()

    if args.model_provider == "openai":
        openai_client = OpenAI()
        model_provider = OpenAIModelProvider(openai_client)
    elif args.model_provider == "anthropic":
        anthropic_client = Anthropic()
        model_provider = AnthropicModelProvider(anthropic_client)
    elif args.model_provider == "ollama":
        model_provider = OllamaModelProvider()
    else:  # transformers
        model_provider = TransformersModelProvider(model_id=args.model_id, device_map=args.transformers_device)

    assistant = Assistant.from_yaml(args.assistant_path)

    # Load previous personas if provided
    previous_personas = []
    if args.previous_personas_path:
        logger.info(f"Loading previous personas from {args.previous_personas_path}")
        conversation_characters = ConversationCharacters.from_yaml(args.previous_personas_path)
        previous_personas = conversation_characters.users
        logger.info(f"Loaded {len(previous_personas)} previous personas")

    # Fixed two-persona generation: victim and perpetrator
    roles = ["victim", "perpetrator"]
    new_personas = []
    for i, role in enumerate(roles):
        persona_generator = PersonaGenerator(
            model_provider,
            args.model_id,
            assistant,
            previous_personas,
            target_role=role,
            scenario_theme=args.scenario,
        )
        print(f"Generating persona {i+1} of 2 (role: {role})")
        persona = persona_generator.generate_persona()
        new_personas.append(persona)
        previous_personas.append(persona)

    # Save only the new personas to the output file
    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    conversation_characters = ConversationCharacters(users=new_personas)
    conversation_characters.to_yaml(args.output_path)
