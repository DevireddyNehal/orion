from datetime import datetime
import json
import os
import time
import concurrent.futures
from typing import Any, Generator
from pathlib import Path

import httpx
from dotenv import load_dotenv
from loguru import logger
from tools.tool_registry import AVAILABLE_TOOL_SCHEMAS, execute_tool

load_dotenv()


class LLMOrchestrator:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "").lower().strip()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

        # If provider not explicitly set, use groq if API key is present, else ollama
        if not self.provider:
            self.provider = "groq" if self.groq_api_key else "ollama"

        self.ollama_url = os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/chat"
        )

        # Default models per provider
        default_model_env = os.getenv("DEFAULT_MODEL")
        if self.provider == "groq":
            self.default_model = default_model_env or "llama-3.3-70b-versatile"
        else:
            self.default_model = default_model_env or "qwen3.5:4b"

        self.groq_client: Any = None
        if self.groq_api_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=self.groq_api_key)
                logger.info(f"[LLM] Groq client initialized with model '{self.default_model}'.")
            except Exception as e:
                logger.warning(f"[LLM] Failed to initialize Groq client: {e}")

        self.history: list[dict] = []

        # Only warm up Ollama if we're using Ollama as primary provider
        if self.provider == "ollama":
            self.warmup_ollama()

    def warmup_ollama(self):
        """Preloads default model into memory with keep_alive so user requests are fast."""
        try:
            logger.info("[LLM] Pre-warming Ollama model '{}' in RAM...", self.default_model)
            t0 = time.time()
            httpx.post(
                self.ollama_url,
                json={
                    "model": self.default_model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": "60m"
                },
                timeout=30.0
            )
            logger.info("[LLM] Warmup complete in {:.2f}s. Model is warm in RAM.", time.time() - t0)
        except Exception as e:
            logger.warning("[LLM] Ollama warmup skipped: {}", e)

    def _trim_history(self, max_messages: int = 10):
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def generate_response_stream(
        self,
        user_text: str,
        target_model: str | None = None,
        prompt: str | None = None
    ) -> Generator[tuple[str, str], None, None]:
        """
        Streams response tokens immediately in real time.
        Autonomously detects and executes tool calls if needed.
        """
        if not target_model:
            target_model = self.default_model

        logger.info(f"[LLM] Provider: {self.provider} | Model: {target_model} | User: {repr(user_text)}")

        self.history.append({
            "role": "user",
            "content": user_text
        })

        now_str = datetime.now().strftime("%A, %B %d, %Y")
        system_content = prompt or (
            f"You are Orion, an intelligent, refined, and articulate voice assistant with the demeanor, calm wit, and polite efficiency of JARVIS.\n"
            f"Always address the user as 'sir'.\n"
            f"Current Date: {now_str}.\n"
            f"Use the current date to determine which events are in the past and which upcoming events are scheduled for the future.\n"
            f"Speak naturally and concisely in 1 to 2 clear sentences.\n"
            f"Never fabricate, guess, or assume real-time facts, current events, recent sports champions, scores, dates, or upcoming schedules.\n"
            f"If the user asks about anything current, recent, or time-sensitive, call the web_search tool once.\n"
            f"When answering from search results, synthesize the facts and state the answer directly in 1 to 2 clear sentences. Never produce evasive filler, speculative monologues, markdown headers, asterisks, bullet points, or lists."
        )

        messages = [
            {
                "role": "system",
                "content": system_content
            },
            *self.history
        ]

        if self.provider == "groq" and self.groq_client is not None:
            try:
                yield from self._stream_groq(list(messages), target_model)
                return
            except Exception as e:
                logger.error(f"[Groq Error] {e}. Falling back to local Ollama...")

        yield from self._stream_ollama(messages, target_model)

    def _stream_groq(self, messages: list[dict], target_model: str) -> Generator[tuple[str, str], None, None]:
        t_start = time.time()
        tool_calls_dict: dict[int, dict] = {}
        turn_content = ""
        has_tool_calls = False
        first_token = True
        final_response_text = ""

        response = self.groq_client.chat.completions.create(  # type: ignore
            model=target_model,
            messages=messages,
            tools=AVAILABLE_TOOL_SCHEMAS,
            stream=True,
            temperature=0.1
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if not delta:
                continue

            if delta.tool_calls:
                has_tool_calls = True
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "type": "function",
                            "function": {"name": tc.function.name or "", "arguments": ""}
                        }
                    if tc.id:
                        tool_calls_dict[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_dict[idx]["function"]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_dict[idx]["function"]["arguments"] += tc.function.arguments

            if delta.content:
                turn_content += delta.content
                if not has_tool_calls:
                    if first_token:
                        logger.info(f"[LLM Groq] Time to first token: {time.time() - t_start:.3f}s")
                        first_token = False
                    final_response_text += delta.content
                    yield delta.content, target_model

        if has_tool_calls and tool_calls_dict:
            raw_calls = list(tool_calls_dict.values())
            logger.info(f"[LLM Groq] Executing {len(raw_calls)} autonomous tool call(s)...")

            def _run_tool_call(call: dict) -> tuple[dict, str]:
                fn = call.get("function", {})
                fn_name = fn.get("name", "")
                fn_args_raw = fn.get("arguments", "{}")
                if isinstance(fn_args_raw, str):
                    try:
                        fn_args = json.loads(fn_args_raw)
                    except Exception:
                        fn_args = {}
                else:
                    fn_args = fn_args_raw or {}

                logger.info(f"[LLM Tool Call] {fn_name}({fn_args})")
                tool_result = execute_tool(fn_name, fn_args)
                return call, tool_result

            results_summary = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(raw_calls), 4)) as executor:
                futures = [executor.submit(_run_tool_call, c) for c in raw_calls]
                for fut in concurrent.futures.as_completed(futures):
                    call, tool_result = fut.result()
                    results_summary.append(tool_result[:1500])

            user_query = ""
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_query = m.get("content", "")
                    break

            now_str = datetime.now().strftime("%A, %B %d, %Y")
            synthesis_messages = [
                {
                    "role": "system",
                    "content": (
                        f"You are Orion, an intelligent, refined voice assistant with the demeanor of JARVIS.\n"
                        f"Always address the user as 'sir'.\n"
                        f"Current Date: {now_str}.\n"
                        f"Speak naturally and concisely in 1 to 2 clear sentences.\n"
                        f"State the facts directly. Never use lists, bullet points, asterisks, or markdown headers."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Search Results:\n"
                        + "\n\n".join(results_summary)
                        + f"\n\nQuestion: {user_query}\n"
                        f"State the final answer directly in 1 to 2 clear sentences:"
                    )
                }
            ]

            final_resp = self.groq_client.chat.completions.create(  # type: ignore
                model=target_model,
                messages=synthesis_messages,
                stream=True,
                temperature=0.1
            )
            first_followup_token = True
            for chunk in final_resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if first_followup_token:
                        logger.info(f"[LLM Groq] Time to first token (after search): {time.time() - t_start:.3f}s")
                        first_followup_token = False
                    final_response_text += delta.content
                    yield delta.content, target_model

        self.history.append({
            "role": "assistant",
            "content": final_response_text
        })
        self._trim_history()

    def _stream_ollama(self, messages: list[dict], target_model: str) -> Generator[tuple[str, str], None, None]:
        payload = {
            "model": target_model if self.provider == "ollama" else "qwen3.5:4b",
            "messages": messages,
            "tools": AVAILABLE_TOOL_SCHEMAS,
            "stream": True,
            "think": False,
            "keep_alive": "60m",
            "options": {
                "temperature": 0.2
            }
        }

        t_start = time.time()
        try:
            first_token = True
            accumulated_content = ""
            accumulated_tool_calls: list[dict] = []

            with httpx.stream("POST", self.ollama_url, json=payload, timeout=60.0) as resp:
                resp.raise_for_status()

                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    msg = chunk.get("message", {})
                    content = msg.get("content", "")
                    tool_calls = msg.get("tool_calls", [])

                    if tool_calls:
                        accumulated_tool_calls.extend(tool_calls)

                    if content and not accumulated_tool_calls:
                        if first_token:
                            logger.info(f"[LLM Ollama] Time to first token: {time.time() - t_start:.3f}s")
                            first_token = False
                        accumulated_content += content
                        yield content, target_model

                    if chunk.get("done"):
                        break

            if accumulated_tool_calls:
                logger.info(f"[LLM Ollama] Executing {len(accumulated_tool_calls)} autonomous tool call(s)...")
                messages.append({
                    "role": "assistant",
                    "content": accumulated_content,
                    "tool_calls": accumulated_tool_calls
                })

                def _run_tool_call(call: dict) -> tuple[dict, str]:
                    fn = call.get("function", {})
                    fn_name = fn.get("name", "")
                    fn_args = fn.get("arguments", {})
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except Exception:
                            fn_args = {}

                    logger.info(f"[LLM Tool Call] {fn_name}({fn_args})")
                    tool_result = execute_tool(fn_name, fn_args)
                    return call, tool_result

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(accumulated_tool_calls), 4)) as executor:
                    futures = [executor.submit(_run_tool_call, c) for c in accumulated_tool_calls]
                    for fut in concurrent.futures.as_completed(futures):
                        _, tool_result = fut.result()
                        messages.append({
                            "role": "tool",
                            "content": tool_result[:1200]
                        })

                stream_payload = {
                    "model": payload["model"],
                    "messages": messages,
                    "stream": True,
                    "think": False,
                    "keep_alive": "60m",
                    "options": {
                        "temperature": 0.1
                    }
                }

                assistant_final_response = ""
                first_followup_token = True

                with httpx.stream("POST", self.ollama_url, json=stream_payload, timeout=60.0) as followup_resp:
                    followup_resp.raise_for_status()
                    for line in followup_resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if first_followup_token:
                            logger.info(f"[LLM Ollama] Time to first token (after tool): {time.time() - t_start:.3f}s")
                            first_followup_token = False

                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            assistant_final_response += content
                            yield content, target_model

                        if chunk.get("done"):
                            break

                self.history.append({
                    "role": "assistant",
                    "content": assistant_final_response
                })
                self._trim_history()
            else:
                self.history.append({
                    "role": "assistant",
                    "content": accumulated_content
                })
                self._trim_history()

        except Exception as e:
            logger.exception(f"[LLM Error] {e}")
            yield f"Error processing request: {e}", "error"