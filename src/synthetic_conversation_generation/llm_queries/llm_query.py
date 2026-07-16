from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
import time
import logging
from typing import Dict

import anthropic
import openai

try:
    import ollama
except ImportError:  # Optional dependency
    ollama = None

try:
    from transformers import pipeline
except ImportError:  # Optional dependency
    pipeline = None


logger = logging.getLogger(__name__)


class LLMQuery(ABC):
    """Base abstract class for LLM queries that defines the common interface."""
    
    @abstractmethod
    def __init__(self, model_provider: ModelProvider, model_id: str):
        self.model_provider = model_provider
        self.model_id = model_id
    
    @abstractmethod
    def generate_prompt(self) -> str:
        """Generate the prompt to send to the LLM."""
        pass
    
    @abstractmethod
    def response_schema(self):
        """Define the expected response schema."""
        pass
    
    @abstractmethod
    def parse_response(self, json_response):
        """Parse the JSON response from the LLM."""
        pass
    
    def query(self, max_retries=3, retry_delay=2, timeout=120):
        """Send the query to the LLM and return the parsed response."""
        user_msg = self.generate_prompt()
        response_schema = self.response_schema()

        retries = 0
        while retries < max_retries:
            try:
                response = self.model_provider.query(user_msg, response_schema, self.model_id, timeout)
                return self.parse_response(response)
            except Exception as e:
                retries += 1
                logger.error(f"Error: {e}")
                logger.info(f"Retrying in {retry_delay} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay += 2

        raise Exception("Unable to complete llm query.")

    
class ModelProvider(ABC):

    @abstractmethod
    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int=60):
        pass

    @abstractmethod
    def response_format(self, response_schema: Dict) -> Dict:
        pass


def _parse_json_from_text(text: str):
    """Best-effort JSON extraction for providers that don't enforce schemas."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(1)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from model response")

class OpenAIModelProvider(ModelProvider):

    def __init__(self, client: openai.OpenAI):
        self.client = client

    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int=60):      

        response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": user_msg}
            ],
            seed=42,
            response_format=self.response_format(response_schema),
            timeout=timeout,
            temperature=1.0
        ).choices[0].message.content

        return json.loads(response)
    
    def response_format(self, response_schema: Dict):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "strict": True,
                "schema": response_schema
            }
        }

class AnthropicModelProvider(ModelProvider):
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int=60):
        """Handle API calls to Anthropic Claude using the tools API for schema enforcement"""
        response_format = self.response_format(response_schema)
        
        response = self.client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_msg}
            ],
            tools=[response_format],
            tool_choice={"type": "tool", "name": response_format["name"]},
            timeout=timeout,
            temperature=1.0
        )
        
        # Parse the response to get the tool use
        for content in response.content:
            if content.type == "tool_use":
                return content.input
        
        # Fallback in case the model didn't use the tool
        raise Exception("Anthropic model did not return a tool use response")
    
    def response_format(self, response_schema: Dict) -> Dict:
        return {
            "name": "json_extractor",
            "description": "Extract structured data according to the provided schema",
            "input_schema": response_schema
        }


class OllamaModelProvider(ModelProvider):
    """
    Ollama-backed provider.

    Note on reasoning models: gpt-oss and similar emit reasoning tokens *before*
    the answer, and that reasoning is charged against the same `num_predict`
    budget as the answer itself. If the budget runs out mid-reasoning, Ollama
    truncates and the answer is never emitted at all -- the response comes back
    with an EMPTY `content` field, which is indistinguishable from a total
    failure unless the `thinking` field is inspected. See `_MAX_OUTPUT_TOKENS`.
    """

    # Maximum tokens the model may generate per call.
    #
    # Was 1024, which silently killed run 6641761 at turn 36: the dialogue-flow
    # planner's prompt had grown to ~2,110 tokens (taxonomy definitions, three
    # planning axes, an exchange budget), the model spent its entire 1024-token
    # allowance reasoning about it, was cut off before emitting any JSON, and
    # returned empty content. Three retries later the pipeline raised.
    #
    # 1024 was set when prompts were short and reasoning models were not in use.
    # The L40S has 44 GiB and the served context is 32,768 tokens, so this cap was
    # leaving almost the whole budget unused. 4096 gives reasoning models room to
    # think *and* answer; unused tokens cost nothing, since generation stops at
    # the closing brace.
    _MAX_OUTPUT_TOKENS = 4096

    def __init__(self, client=None, max_output_tokens: int | None = None):
        if client is None:
            if ollama is None:
                raise ImportError("ollama package not installed. Install with `pip install ollama`." )
            client = ollama
        self.client = client
        self.max_output_tokens = max_output_tokens or self._MAX_OUTPUT_TOKENS

    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int = 60):
        system_prompt = (
            "You are a JSON API. Return ONLY valid JSON matching this schema. "
            "Do not include explanations or extra text. Schema: "
            f"{json.dumps(response_schema)}"
        )

        response = self.client.chat(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            format="json",
            options={"temperature": 0.6, "num_predict": self.max_output_tokens},
            stream=False,
        )

        message = response.get("message", {}) or {}
        content = message.get("content", "") or ""

        # Reasoning models return their chain of thought separately. If content is
        # empty but thinking is not, the model was truncated mid-reasoning and
        # never reached the answer -- surface that explicitly rather than
        # reporting a bare "parse failed" with nothing to go on.
        thinking = message.get("thinking", "") or ""

        try:
            return _parse_json_from_text(content)
        except Exception as exc:
            raise ValueError(self._diagnose(content, thinking, response)) from exc

    def _diagnose(self, content: str, thinking: str, response: Dict) -> str:
        """Build an error message that identifies *why* the response was unusable."""
        eval_count = response.get("eval_count")
        parts = ["Ollama response parse failed."]

        if not content and thinking:
            parts.append(
                f"Content was EMPTY but the model produced {len(thinking)} chars of "
                f"reasoning — it was truncated mid-thought and never emitted an answer. "
                f"num_predict={self.max_output_tokens} is likely too low for this prompt."
            )
        elif not content:
            parts.append("Content was EMPTY and no reasoning was returned.")
        else:
            parts.append(f"Content ({len(content)} chars): {content[:400]}")

        if eval_count is not None:
            hit_cap = eval_count >= self.max_output_tokens
            parts.append(
                f"Generated {eval_count} tokens (cap {self.max_output_tokens})"
                + (" — HIT THE CAP, so the output is truncated." if hit_cap else ".")
            )
        if thinking:
            parts.append(f"Reasoning tail: ...{thinking[-200:]}")
        return " ".join(parts)

    def response_format(self, response_schema: Dict) -> Dict:
        return response_schema


class TransformersModelProvider(ModelProvider):

    def __init__(self, model_id: str, device_map: str = "auto", max_new_tokens: int = 512, temperature: float = 0.8, **generator_kwargs):
        if pipeline is None:
            raise ImportError("transformers package not installed. Install with `pip install transformers`.")

        # Lazy-load text-generation pipeline for local models
        self.generator = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=model_id,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.generation_args = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "return_full_text": False,
        }
        self.generation_args.update(generator_kwargs)

    def query(self, user_msg: str, response_schema: Dict, model_id: str, timeout: int = 60):
        prompt = (
            "You are a JSON API. Reply with ONLY a JSON object that conforms to this schema: "
            f"{json.dumps(response_schema)}\n"
            "If you cannot, return an empty JSON object {}."
        )
        full_prompt = f"{prompt}\nUser request:\n{user_msg}\nJSON:"  

        result = self.generator(full_prompt, **self.generation_args)[0]["generated_text"]
        return _parse_json_from_text(result)

    def response_format(self, response_schema: Dict) -> Dict:
        return response_schema
