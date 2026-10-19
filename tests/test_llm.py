import json
import unittest
from unittest.mock import MagicMock, patch

from llm.llm import LLMOrchestrator


class TestLLMOrchestrator(unittest.TestCase):
    def test_initialization_defaults_ollama_when_no_key(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "", "LLM_PROVIDER": ""}, clear=True):
            with patch.object(LLMOrchestrator, "warmup_ollama"):
                orchestrator = LLMOrchestrator()
                self.assertEqual(orchestrator.provider, "ollama")
                self.assertEqual(orchestrator.default_model, "qwen3.5:4b")

    def test_initialization_defaults_groq_when_key_present(self):
        with patch.dict(
            "os.environ",
            {"GROQ_API_KEY": "gsk_dummy123", "LLM_PROVIDER": ""},
            clear=True,
        ):
            with patch("groq.Groq") as mock_groq_cls:
                orchestrator = LLMOrchestrator()
                self.assertEqual(orchestrator.provider, "groq")
                self.assertEqual(orchestrator.default_model, "llama-3.3-70b-versatile")
                mock_groq_cls.assert_called_once_with(api_key="gsk_dummy123")

    def test_trim_history(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "", "LLM_PROVIDER": ""}, clear=True):
            with patch.object(LLMOrchestrator, "warmup_ollama"):
                orchestrator = LLMOrchestrator()
                orchestrator.history = [{"role": "user", "content": f"msg {i}"} for i in range(15)]
                self.assertEqual(len(orchestrator.history), 15)

                orchestrator._trim_history(max_messages=10)
                self.assertEqual(len(orchestrator.history), 10)
                self.assertEqual(orchestrator.history[0]["content"], "msg 5")
                self.assertEqual(orchestrator.history[-1]["content"], "msg 14")

    def test_stream_ollama_basic(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "", "LLM_PROVIDER": "ollama"}, clear=True):
            with patch.object(LLMOrchestrator, "warmup_ollama"):
                orchestrator = LLMOrchestrator()

        mock_chunks = [
            json.dumps({"message": {"content": "Hello", "role": "assistant"}, "done": False}),
            json.dumps({"message": {"content": " sir!", "role": "assistant"}, "done": False}),
            json.dumps({"done": True}),
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = mock_chunks
        mock_resp.__enter__.return_value = mock_resp

        with patch("httpx.stream", return_value=mock_resp):
            stream = orchestrator._stream_ollama(
                messages=[{"role": "user", "content": "hello"}],
                target_model="qwen3.5:4b",
            )
            tokens = [token for token, model in stream]

        self.assertEqual("".join(tokens), "Hello sir!")
        self.assertEqual(len(orchestrator.history), 1)
        self.assertEqual(orchestrator.history[0]["content"], "Hello sir!")

    def test_stream_ollama_generator_exit(self):
        with patch.dict("os.environ", {"GROQ_API_KEY": "", "LLM_PROVIDER": "ollama"}, clear=True):
            with patch.object(LLMOrchestrator, "warmup_ollama"):
                orchestrator = LLMOrchestrator()

        def chunk_gen():
            yield json.dumps({"message": {"content": "Part one", "role": "assistant"}, "done": False})
            yield json.dumps({"message": {"content": " Part two", "role": "assistant"}, "done": False})

        mock_resp = MagicMock()
        mock_resp.iter_lines.side_effect = chunk_gen
        mock_resp.__enter__.return_value = mock_resp

        with patch("httpx.stream", return_value=mock_resp):
            stream = orchestrator._stream_ollama(
                messages=[{"role": "user", "content": "tell me a story"}],
                target_model="qwen3.5:4b",
            )
            first_token = next(stream)
            self.assertEqual(first_token, ("Part one", "qwen3.5:4b"))
            # Abort stream (user interrupts / barge-in)
            stream.close()

        self.assertEqual(len(orchestrator.history), 1)
        self.assertIn("[interrupted]", orchestrator.history[0]["content"])
        self.assertIn("Part one", orchestrator.history[0]["content"])

    def test_stream_ollama_connection_error(self):
        import httpx

        with patch.dict("os.environ", {"GROQ_API_KEY": "", "LLM_PROVIDER": "ollama"}, clear=True):
            with patch.object(LLMOrchestrator, "warmup_ollama"):
                orchestrator = LLMOrchestrator()

        with patch("httpx.stream", side_effect=httpx.ConnectError("Connection refused")):
            stream = orchestrator._stream_ollama(
                messages=[{"role": "user", "content": "hi"}],
                target_model="qwen3.5:4b",
            )
            results = list(stream)

        self.assertEqual(len(results), 1)
        self.assertIn("trouble connecting", results[0][0])
        self.assertEqual(results[0][1], "error")

    def test_groq_fallback_to_ollama(self):
        with patch.dict(
            "os.environ",
            {"GROQ_API_KEY": "gsk_key", "LLM_PROVIDER": "groq"},
            clear=True,
        ):
            with patch("groq.Groq") as mock_groq_cls:
                mock_client = MagicMock()
                mock_groq_cls.return_value = mock_client
                orchestrator = LLMOrchestrator()

        # Simulate Groq error in _stream_groq
        with patch.object(
            orchestrator,
            "_stream_groq",
            side_effect=RuntimeError("Groq API rate limit or outage"),
        ):
            with patch.object(
                orchestrator,
                "_stream_ollama",
                return_value=iter([("Fallback response", "qwen3.5:4b")]),
            ) as mock_ollama:
                stream = orchestrator.generate_response_stream("status report")
                results = list(stream)

                mock_ollama.assert_called_once()
                self.assertEqual(results, [("Fallback response", "qwen3.5:4b")])


if __name__ == "__main__":
    unittest.main()
