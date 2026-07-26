import json
import os
import time
from typing import Generator

import httpx
from dotenv import load_dotenv
from loguru import logger

from models import Route

load_dotenv()


class LLMOrchestrator:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL")

        self.intent_model = os.getenv(
            "INTENT_MODEL",
            "qwen3.5:0.8b"
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

        self.history = []

    def intent_classifier(self, user_text: str) -> Route:
        """
        Placeholder intent classifier.
        Later this will call a small routing model.
        """

        return Route(
            model=self.intent_model,
            capabilities=["obsidian"]
        )

    def generate_response(
        self,
        user_text: str,
        target_model: str
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
                    "content": self.response_system_prompt
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

                assistant_response = ""
                first = True

                for line in r.iter_lines():
                    print(repr(line))

                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if first:
                        logger.info(
                            "Time to first token: %.3fs",
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

            MAX_MESSAGES = 20

            if len(self.history) > MAX_MESSAGES:
                self.history = self.history[-MAX_MESSAGES:]

        except Exception as e:
            logger.error(e)
            yield f"Error: {e}", "error"