import json
import os
import time
from typing import Generator

import httpx
from dotenv import load_dotenv
from loguru import logger
from dataclasses import dataclass
from pathlib import Path

load_dotenv()

@dataclass
class Route:
    model: str
    capabilities: list[str]


class LLMOrchestrator:
    def __init__(self):

        BASE_DIR = Path(__file__).parent
        
        self.ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/chat"
        )

        self.intent_model = os.getenv(
            "INTENT_MODEL",
            "qwen3.5:0.8b"
        )

        self.default_model = os.getenv(
            "DEFAULT_MODEL",
            "qwen3.5:4b"
        )

        self.response_system_prompt = (
            "You are a voice assistant. "
            "Respond instantly, concisely, and conversationally. "
            "Do not use think tags or reasoning chains. "
            "Go straight to the answer."
            "Respond naturally as spoken English."
            "Do not use emojis."
            "Do not use markdown."
            "Do not use stage directions or roleplay."
            "Keep responses concise."
        )

        self.intent_system_prompt = Path(BASE_DIR / "intent_system_prompt.txt").read_text()

        self.history = []

    def _trim_history(self, max_messages: int = 20):
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def intent_classifier(self, user_text: str) -> Route:
        logger.info("Intent model: {}", self.intent_model)

        logger.info("Sending request to intent classifier")
        route_response = self.generate_response(
            user_text,
            self.intent_model,
            prompt=self.intent_system_prompt,
            use_history=False
        )

        logger.info("Intent classifier response: {}", route_response)

        try:
            data = json.loads(route_response)

            model = data.get("model")
            capabilities = data.get("capabilities")

            if (
                isinstance(model, str)
                and model.strip()
                and isinstance(capabilities, list)
                and all(isinstance(c, str) for c in capabilities)
            ):
                return Route(
                    model=model,
                    capabilities=capabilities,
                )

            logger.error("Invalid route JSON structure: {}", data)

        except (json.JSONDecodeError, TypeError) as e:
            logger.error("Failed to parse intent classifier response: {}", e)

        # Safe fallback
        return Route(
            model=self.default_model,
            capabilities=[],
        )

    def generate_response(
        self,
        user_text: str,
        target_model: str,
        prompt: str | None = None,
        use_history: bool = True
    ) -> str:
        """Generates a response from the LLM for a given user input."""
        logger.info(f"Model: {target_model}")

        if use_history:
            self.history.append({
                "role": "user",
                "content": user_text
            })

        messages = [
            {
                "role": "system",
                "content": prompt
            }
        ]

        if use_history:
            messages.extend(self.history)

        messages.append({
            "role": "user",
            "content": user_text
        })

        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "think": False
        }

        try:
            response = httpx.post(
                self.ollama_url,
                json=payload,
                timeout=120.0
            )

            response.raise_for_status()
            data = response.json()

            assistant_response = data.get("message", {}).get("content", "")

            if not assistant_response:
                logger.warning("Unexpected response: {}", data)

            if use_history:
                self.history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                self._trim_history()

            return assistant_response

        except Exception as e:
            logger.error(e)
            return f"Error: {e}"

    def generate_response_stream(
        self,
        user_text: str,
        target_model: str,
        prompt: str | None = None
    ) -> Generator[tuple[str, str], None, None]:

        logger.info(f"Model: {target_model}")

        self.history.append({
            "role": "user",
            "content": user_text
        })

        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "system",
                    "content": prompt
                },
                *self.history
            ],
            "stream": True,
            "think": False
        }

        t_request = time.time()
        try:
            with httpx.stream(
                "POST",
                self.ollama_url,
                json=payload,
                timeout=120.0
            ) as r:

                r.raise_for_status()

                assistant_response = ""
                first = True

                for line in r.iter_lines():

                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if first:
                        logger.info(
                            "Time to first token: {:.3f}s",
                            time.time() - t_request
                        )
                        first = False

                    content = chunk.get("message", {}).get("content", "")

                    if content:
                        assistant_response += content
                        yield content, target_model

                    if chunk.get("done"):
                        break

            self.history.append({
                "role": "assistant",
                "content": assistant_response
            })
            self._trim_history()

        except Exception as e:
            logger.error(e)
            yield f"Error: {e}", "error"