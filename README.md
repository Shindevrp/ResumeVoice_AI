# ResumeVoice AI

<p align="center">
  <img src="assets/resumevoice-cover.svg" alt="ResumeVoice AI — your personal voice agent" width="700">
</p>

A real-time conversational voice agent that speaks as *you*. Load a resume into `resume/` and the agent answers interview questions in first person — your personal AI voice avatar.

It is a full speech pipeline, not a chatbot: it **listens** continuously, **understands** topic/intent/emotion, **thinks** with a streaming LLM, and **speaks** with natural timing, interruptions, and backchannels.

## Quick Start (local, no GPU)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --editable ".[all]"

# LLM via Ollama
ollama pull qwen3:8b
export RESUMEVOICE_LLM_URL=http://localhost:11434/v1
export RESUMEVOICE_LLM_MODEL=qwen3:8b

# CPU machine: keep STT/VAD off cuda
export RESUMEVOICE_STT_DEVICE=cpu
export RESUMEVOICE_STT_COMPUTE=int8
export RESUMEVOICE_VAD_DEVICE=cpu

python3 -m app.cli server
```

Open `http://localhost:8000/ui` (WebSocket voice UI), `http://localhost:8000/webrtc`, or `http://localhost:8501` for the live dashboard (Streamlit).

Docker (GPU, vLLM + app): `docker compose up` — app on `:8000`, LLM on `:8001`.

## Resume Persona

The agent speaks as the person in `resume/resume.txt` (or any `.txt`/`.md`/`.pdf` you point at):

| Env var | Default | Description |
|---|---|---|
| `RESUMEVOICE_RESUME_ENABLED` | `1` | Set `0` to disable the persona |
| `RESUMEVOICE_RESUME_PATH` | `resume/resume.txt` | Path to resume text or PDF |

How it works:
- A **persona block** (first-person instructions + name/headline/contact/summary) is injected into every turn's system prompt.
- Resume **sections** (experience, skills, education, projects, publications…) are seeded into each session's vector store, so detailed questions retrieve the exact section.
- The agent stays in character — it never says "according to my resume" or breaks into an AI assistant.
- Replace the file (or drop a new PDF) and restart; `pypdf` (extra: `.[resume]`) extracts PDFs and caches them as `.txt`.

## How It Works

```
LISTEN ──► THINK ──► SPEAK
audio ─ VAD ─ partial STT ─ turn detector
      ─ final STT ─ topic / intent / emotion
      ─ speculative LLM (starts on the partial transcript)
      ─ sentence chunker + prosody → Piper TTS → speaker
```

- **Turn-taking** — silence + prosody + linguistic cues, with per-turn adaptive thresholds.
- **Barge-in** — the user can interrupt speech or LLM generation; the pipeline cancels and listens.
- **Backchannels** — "uh-huh", "hmm", "right" on natural pauses, plus thinking fillers.
- **Pacing** — human-like 50–600ms response delay based on engagement.
- **Memory** — 20-turn window, rolling LLM summary, topic-keyed vector retrieval, durable facts (regex + LLM), all per-session.
- **Tools** — inline `{tool:name(args)}` calls (`get_time`, `calculate`, `roll_dice`, …) stop TTS, execute, and continue naturally.

## Project Structure

```
├── app/                 # FastAPI server, routes, CLI, client UIs
│   ├── server.py        # app wiring, provider build, resume load, warm-ups
│   ├── cli.py           # `server` / `demo` commands
│   ├── session_registry.py
│   ├── routes/          # ws, webrtc, chat, sessions, metrics, health
│   └── ui.html · mic.html · webrtc.html
├── core/                # pipeline.py (StreamingPipeline), state.py (FSM), config.py
├── modules/             # vad, turn, prosody, emotion, memory, tts, tools, metrics, backchannel, dialogue
│   └── dialogue/        # prompts.py, resume.py (persona loader)
├── providers/           # stt (faster-whisper), llm (vLLM/Ollama), tts (Piper)
├── resume/              # resume.txt — the persona knowledge file
├── utils/               # audio, logger
├── tests/               # 190 unit tests
├── streamlit_app.py     # live dashboard
├── Dockerfile · docker-compose.yml
└── pyproject.toml
```

## HTTP API

| Endpoint | Method | Description |
|---|---|---|
| `/health`, `/health/ready`, `/health/live` | GET | Health/readiness |
| `/metrics` | GET | Uptime + per-stage latency |
| `/sessions` | GET | Live session snapshots |
| `/chat/`, `/chat/stream` | POST | One-shot / SSE text chat |
| `/ui`, `/mic`, `/webrtc` | GET | Client HTML |
| `/ws/audio` | WS | WebSocket PCM 16kHz audio |
| `/ws/signal` | WS | WebRTC signaling (Opus RTP) |

## Configuration

Core env vars (see `core/config.py` and `.env.example` for the full list):

`RESUMEVOICE_STT_MODEL` · `RESUMEVOICE_STT_DEVICE` · `RESUMEVOICE_LLM_URL` · `RESUMEVOICE_LLM_MODEL` · `RESUMEVOICE_LLM_API_KEY` · `RESUMEVOICE_TTS_MODEL` · `RESUMEVOICE_VAD_THRESHOLD` · `RESUMEVOICE_EMOTION_ENABLED` · `RESUMEVOICE_RESUME_ENABLED` · `RESUMEVOICE_RESUME_PATH`

## Testing

```bash
pytest tests/ -v   # 190 passed
```

## Stack

Silero VAD · faster-whisper (CTranslate2) · vLLM / Ollama (OpenAI-compatible) · Piper TTS · FastAPI + Uvicorn · aiortc (WebRTC) · sentence-transformers (retrieval) · Streamlit (dashboard) · Docker Compose.
