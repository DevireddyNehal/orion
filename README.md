# 🌌 Orion

> **An ultra-low-latency, real-time AI voice assistant and personal Chief of Staff.**

Orion is engineered for natural, fluid human-AI voice conversations. Designed to run both locally and with high-speed cloud acceleration, Orion achieves sub-second turn times (<1s) by pipelining speech-to-text, streaming token generation, and real-time neural audio synthesis with instant barge-in interrupt capability.

---

## ⚡ Key Highlights

- **Sub-Second Turnaround Latency (<1s)**: Overlapping pipeline where LLM token streaming feeds directly into sentence-level TTS audio chunking.
- **Barge-In / Instant Interrupt**: Interrupt Orion mid-sentence. Triggering recording instantly cuts audio playback and frees the audio device with zero deadlocks.
- **Custom Neural Voice Blending**: Powered by Kokoro TTS with custom voice weight blends (e.g., combining British crispness and warm conversational tones), alongside Piper ONNX and Edge-TTS backends.
- **Dual Cloud & Local Hybrid Engine**:
  - **Cloud Speed**: Groq LLM (`llama-3.3-70b-versatile`) + Groq STT (`whisper-large-v3-turbo`) for blindingly fast responses (<0.3s).
  - **Local Privacy**: Offline fallback with Ollama (`qwen2.5` / `llama3.2`) and local `faster-whisper`.
- **Whisper Hallucination Filtering**: Cleans out common silence artifacts (*"Thanks for watching"*, blank audio clips) before sending queries downstream.
- **Native OS Tool Calling**:
  - ⏰ **GNOME Clocks**: Automatically creates and manages alarms and timers directly via Linux `gsettings`.
  - 🔍 **Web Search**: Dynamic real-time internet searches via DuckDuckGo, Tavily, or SerpAPI.
  - 🔔 **Voice Reminders**: Background scheduled reminders that trigger voice notifications when due.

---

## 🏗️ Architecture Pipeline

```
  User Speaks ──► [ AudioRecorder ] (SoundDevice)
                         │
                         ▼
                  [ Whisper STT ] (Groq Cloud / Faster-Whisper Local)
                         │
                         ▼
               [ LLM Orchestrator ] (Groq Llama 3.3 / Local Ollama)
                  │              │
        Tool Call ▼              ▼ Streaming Tokens
      [ Tools Registry ]   [ Punctuation / Clause Chunking ]
      (Clocks, Web, Reminders)   │
                                 ▼
                         [ Streaming TTS ] (Kokoro Blend / Piper / Edge)
                                 │
                                 ▼
                         Speaker Audio Output
```

---

## 🚀 Quickstart

Orion uses [`uv`](https://github.com/astral-sh/uv) for lightning-fast Python dependency management.

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Orion.git
cd Orion
```

### 2. Configure Environment Variables
Copy the template and fill in your preferred providers and API keys:
```bash
cp .env.example .env
```

Minimal keys needed (depending on chosen providers):
- `GROQ_API_KEY`: For ultra-fast cloud LLM and STT ([Groq Console](https://console.groq.com/))
- `TAVILY_API_KEY`: Optional, for live web search

### 3. Install Dependencies
```bash
uv sync
```

### 4. Run Orion
```bash
uv run main.py
```

---

## ⚙️ Configuration

All runtime behavior is controlled via `.env`. Key options include:

| Setting | Options | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `groq` \| `ollama` | LLM backend (cloud ultra-fast or local offline) |
| `DEFAULT_MODEL` | `llama-3.3-70b-versatile`, `qwen2.5`, etc. | Model name for the LLM provider |
| `STT_PROVIDER` | `groq` \| `local` | Speech recognition backend |
| `STT_MODEL_SIZE`| `base.en`, `small.en`, `large-v3` | Local Faster-Whisper model size |
| `TTS_PROVIDER` | `kokoro` \| `piper` \| `edge_tts` | Text-to-speech synthesis engine |
| `TTS_VOICE` | `orion`, `bm_lewis`, custom blend | Voice preset or weighted blend |
| `SEARCH_PROVIDER`| `duckduckgo` \| `tavily` \| `serp` | Search tool engine |

---

## 🧩 Project Structure

```
Orion/
├── main.py              # Main event loop, barge-in binding, & audio pipeline
├── llm/
│   ├── llm.py           # LLM Orchestrator, multi-turn history & tool routing
│   └── intent_system_prompt.txt
├── voice/
│   ├── recorder.py      # Audio input capture & keypress callbacks
│   ├── stt.py           # Speech-to-text (Groq / Faster-Whisper)
│   ├── tts.py           # Streaming TTS playback & interrupt controls
│   └── providers.py     # Kokoro, Piper, & EdgeTTS synthesis backends
├── tools/
│   ├── tool_registry.py # Function calling interface
│   ├── gnome_clock.py   # Linux GNOME Clocks timer/alarm automation
│   ├── reminders.py     # Scheduled background voice reminders
│   └── web_search.py    # Search providers (DuckDuckGo, Tavily, SerpAPI)
├── .env.example         # Documented template for all settings
└── pyproject.toml       # Project metadata & dependencies (uv managed)
```

---

## 🛣️ Roadmap

- [x] Sub-second streaming conversational loop
- [x] Barge-in audio interruption
- [x] Linux system tool integration (GNOME Clocks)
- [x] Background voice reminder system
- [ ] Multi-turn persistent memory & vector world-model
- [ ] Vision integration (desktop / webcam screen context)
- [ ] Calendar & email personal assistant integrations

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
