"""
Preset stack combos for the precall config page.

Each combo is a (label, stt, llm, tts) tuple where the keys are what the
LiveKit Agent's plugin builder understands:

    STT keys: deepgram-nova-3 | deepgram-nova-2
              | groq-whisper-large-v3-turbo (blocked on some networks)
              | local-whisper-tiny | local-whisper-base
    LLM keys: gemini-gemini-2.0-flash | gemini-gemini-1.5-flash
              | ollama-llama3.1:8b-instruct-q4_K_M | ollama-llama3.2:3b
              | groq-llama-3.1-8b-instant (blocked on some networks)
    TTS keys: elevenlabs-eleven_turbo_v2_5 | elevenlabs-eleven_flash_v2_5
              | piper-en_US-lessac-medium | kokoro-af_bella
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Combo:
    key: str          # internal id, sent to the agent in job metadata
    label: str        # human label shown in the picker
    description: str  # one-line caption
    stt: str          # plugin key for STT
    llm: str          # plugin key for LLM
    tts: str          # plugin key for TTS
    badge: str = ""   # optional pill


# First combo is the default — picked when no combo/stt/llm/tts is specified.
PRESETS: list[Combo] = [
    Combo(
        key="deepgram-gemini-elevenlabs",
        label="Deepgram + Gemini + ElevenLabs",
        description="Nova-3 STT · Gemini 2.0 Flash · ElevenLabs Turbo. Best cloud quality, works without Groq.",
        stt="deepgram-nova-3",
        llm="gemini-gemini-2.0-flash",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="default",
    ),
    Combo(
        key="deepgram-ollama-elevenlabs",
        label="Deepgram + Ollama + ElevenLabs",
        description="Nova-3 STT · Llama 3.1 8B local · ElevenLabs Turbo. No cloud LLM — Ollama must be running.",
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="no cloud LLM",
    ),
    Combo(
        key="deepgram-gemini-elevenlabs-flash",
        label="Deepgram + Gemini 1.5 + ElevenLabs Flash",
        description="Nova-3 STT · Gemini 1.5 Flash · ElevenLabs Flash v2.5. Ultra-low latency.",
        stt="deepgram-nova-3",
        llm="gemini-gemini-1.5-flash",
        tts="elevenlabs-eleven_flash_v2_5",
        badge="low latency",
    ),
    Combo(
        key="deepgram-together-elevenlabs",
        label="Deepgram + Together.ai + ElevenLabs",
        description="Nova-3 STT · Llama 3.1 8B cloud GPU · ElevenLabs Turbo. Groq-speed, no Groq dependency.",
        stt="deepgram-nova-3",
        llm="together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="cloud fast",
    ),
    Combo(
        key="deepgram-openrouter-elevenlabs",
        label="Deepgram + OpenRouter + ElevenLabs",
        description="Nova-3 STT · Llama 3.1 8B via OpenRouter (free models available) · ElevenLabs Turbo.",
        stt="deepgram-nova-3",
        llm="openrouter-meta-llama/llama-3.1-8b-instruct:free",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="free LLM",
    ),
    Combo(
        key="groq-elevenlabs",
        label="Groq + ElevenLabs",
        description="Whisper STT · Llama 3.1 8B · ElevenLabs Turbo. Requires Groq API (may be blocked on some networks).",
        stt="groq-whisper-large-v3-turbo",
        llm="groq-llama-3.1-8b-instant",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="groq",
    ),
    Combo(
        key="deepgram-elevenlabs",
        label="Deepgram + Groq LLM + ElevenLabs",
        description="Nova-3 STT · Llama 3.1 8B (Groq) · ElevenLabs Turbo. Best Indian English STT.",
        stt="deepgram-nova-3",
        llm="groq-llama-3.1-8b-instant",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="best Indian STT",
    ),
    Combo(
        key="local-oss",
        label="Local OSS (fully offline)",
        description="Whisper tiny · Qwen2.5 0.5B (Ollama) · Piper. Zero cloud calls, real-time on CPU.",
        stt="local-whisper-tiny",
        llm="ollama-qwen2.5:0.5b-instruct",
        tts="piper-en_US-lessac-medium",
        badge="offline",
    ),
]


STT_OPTIONS = [
    {"key": "deepgram-nova-3",             "label": "Deepgram · Nova-3  (best Indian English)",
     "group": "Cloud — Deepgram"},
    {"key": "deepgram-nova-2",             "label": "Deepgram · Nova-2",
     "group": "Cloud — Deepgram"},
    {"key": "groq-whisper-large-v3-turbo", "label": "Groq · Whisper large-v3-turbo",
     "group": "Cloud — Groq"},
    {"key": "groq-whisper-large-v3",       "label": "Groq · Whisper large-v3  (max accuracy)",
     "group": "Cloud — Groq"},
    {"key": "local-whisper-tiny",          "label": "Local · Whisper tiny  (CPU, offline)",
     "group": "Local — Whisper"},
    {"key": "local-whisper-base",          "label": "Local · Whisper base  (slower, more accurate)",
     "group": "Local — Whisper"},
]

LLM_OPTIONS = [
    {"key": "gemini-gemini-2.0-flash",             "label": "Gemini · 2.0 Flash  (free, fast)",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-1.5-flash",             "label": "Gemini · 1.5 Flash  (free, stable)",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-2.0-flash-lite",        "label": "Gemini · 2.0 Flash Lite  (cheapest)",
     "group": "Cloud — Google"},
    {"key": "together-meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                                                   "label": "Together.ai · Llama 3.1 8B Turbo  (cloud GPU, fast)",
     "group": "Cloud — Together.ai"},
    {"key": "together-meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                                                   "label": "Together.ai · Llama 3.1 70B Turbo  (highest quality)",
     "group": "Cloud — Together.ai"},
    {"key": "openrouter-meta-llama/llama-3.1-8b-instruct:free",
                                                   "label": "OpenRouter · Llama 3.1 8B  (free tier)",
     "group": "Cloud — OpenRouter"},
    {"key": "openrouter-google/gemini-2.0-flash-exp:free",
                                                   "label": "OpenRouter · Gemini 2.0 Flash  (free tier)",
     "group": "Cloud — OpenRouter"},
    {"key": "groq-llama-3.1-8b-instant",           "label": "Groq · Llama 3.1 8B Instant",
     "group": "Cloud — Groq"},
    {"key": "groq-llama-3.3-70b-versatile",        "label": "Groq · Llama 3.3 70B Versatile",
     "group": "Cloud — Groq"},
    {"key": "ollama-llama3.1:8b-instruct-q4_K_M",  "label": "Ollama · Llama 3.1 8B  (best local)",
     "group": "Local — Ollama"},
    {"key": "ollama-llama3.2:3b",                  "label": "Ollama · Llama 3.2 3B  (fast)",
     "group": "Local — Ollama"},
    {"key": "ollama-llama3.2:1b",                  "label": "Ollama · Llama 3.2 1B  (fastest)",
     "group": "Local — Ollama"},
    {"key": "ollama-qwen2.5:0.5b-instruct",        "label": "Ollama · Qwen2.5 0.5B  (tiny)",
     "group": "Local — Ollama"},
    {"key": "ollama-qwen2.5:1.5b-instruct",        "label": "Ollama · Qwen2.5 1.5B",
     "group": "Local — Ollama"},
]

TTS_OPTIONS = [
    {"key": "elevenlabs-eleven_turbo_v2_5",      "label": "ElevenLabs · Turbo v2.5  (best quality)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",      "label": "ElevenLabs · Flash v2.5  (lowest latency)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_multilingual_v2", "label": "ElevenLabs · Multilingual v2 (29 languages)",
     "group": "Cloud — ElevenLabs"},
    {"key": "cartesia-sonic-3",                  "label": "Cartesia · Sonic-3  (natural voice)",
     "group": "Cloud — Cartesia"},
    {"key": "cartesia-sonic-turbo",              "label": "Cartesia · Sonic Turbo  (lowest latency)",
     "group": "Cloud — Cartesia"},
    {"key": "piper-en_US-lessac-medium",         "label": "Piper · lessac  (US female, CPU offline)",
     "group": "Local — Piper"},
    {"key": "piper-en_US-ryan-high",             "label": "Piper · ryan-high  (US male)",
     "group": "Local — Piper"},
    {"key": "piper-en_GB-alan-medium",           "label": "Piper · alan-medium  (British male)",
     "group": "Local — Piper"},
    {"key": "kokoro-af_bella",                   "label": "Kokoro · af_bella  (warm female, ~150 MB)",
     "group": "Local — Kokoro"},
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
