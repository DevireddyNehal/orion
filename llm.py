import json
from loguru import logger
import time
import httpx
from typing import Generator

class LLMOrchestrator:
    def __init__(self):
        self.fast_local_model = "qwen3.5:0.8b"
        self.ollama_url = "http://localhost:11434/api/chat"
        self.response_system_prompt = "You are a voice assistant. Respond instantly, concisely, and conversationally. Do not use think tags or reasoning chains. Go straight to the answer."
        self.history = []

    def generate_response(self, user_text: str) -> Generator[tuple[str, str], None, None]:
        """Routes and executes the user query using a direct HTTP stream to bypass library stalls."""
        target_model = "qwen3.5:4b"
        logger.info(f"Model: {target_model}")

        self.history.append({
            "role": "user",
            "content": user_text
        })
        
        # T0: Build payload
        t_start = time.time()
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
        logger.info(f"T0 Build payload: {time.time() - t_start:.4f}s")

        try:
            # T1: Opening HTTP connection
            t_conn_start = time.time()
            with httpx.stream("POST", self.ollama_url, json=payload, timeout=120.0) as r:
                logger.info(f"T1 Opening HTTP connection: {time.time() - t_conn_start:.4f}s")
                
                # T2: HTTP connection opened (Headers received)
                logger.info(f"T2 HTTP connection opened: {time.time() - t_conn_start:.4f}s")
                
                first = True
                # T3: Reading first line
                assistant_response = ""
                t_line_start = time.time()
                for line in r.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON Decode Error: {e} | Line: {line}")
                        continue

                    if first:
                        logger.info(f"T3 First line received: {time.time() - t_line_start:.4f}s")
                        t_parse_start = time.time()
                        logger.info(f"T4 First JSON parsed: {time.time() - t_parse_start:.4f}s")
                        first = False

                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        if not first: # First token is handled by the block above, so we just yield
                            pass 
                        
                        # Log T5 only for the very first content found
                        if 't_yield_start' not in locals():
                            t_yield_start = time.time()
                            logger.info(f"T5 First token yielded: {t_yield_start - t_line_start:.4f}s")
                        
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
            logger.error(f"HTTP Stream Error: {e}")
            yield f"Error during streaming execution: {str(e)}", "error"