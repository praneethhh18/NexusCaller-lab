"""
Preset stack combos for the precall config page.

Each combo is a (label, stt, llm, tts) tuple where the keys are what the
LiveKit Agent's plugin builder understands:

    STT keys: deepgram-nova-3 | deepgram-nova-2 | groq-whisper-large-v3-turbo
              | local-whisper-tiny | local-whisper-base
    LLM keys: groq-llama-3.1-8b-instant | groq-llama-3.3-70b-versatile
              | groq-openai/gpt-oss-120b | ollama-qwen2.5:0.5b-instruct
    TTS keys: elevenlabs-eleven_turbo_v2_5 | elevenlabs-eleven_flash_v2_5
              | cartesia-sonic-3 | cartesia-sonic-turbo
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
    badge: str = ""   # optional pill ("default", "best for Indian English", …)


PRESETS: list[Combo] = [
    Combo(
        key="groq-elevenlabs",
        label="Groq + ElevenLabs",
        description="Whisper STT · Llama 3.1 8B · ElevenLabs Turbo. Free, no monthly cap.",
        stt="groq-whisper-large-v3-turbo",
        llm="groq-llama-3.1-8b-instant",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="default",
    ),
    Combo(
        key="deepgram-elevenlabs",
        label="Deepgram + ElevenLabs",
        description="Nova-3 STT · Llama 3.1 8B · ElevenLabs Turbo. Best Indian English transcription.",
        stt="deepgram-nova-3",
        llm="groq-llama-3.1-8b-instant",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="best for Indian English",
    ),
    Combo(
        key="deepgram-cartesia",
        label="Deepgram + Cartesia",
        description="Nova-3 STT · Llama 3.1 8B · Cartesia Sonic-3. Most natural-sounding voice.",
        stt="deepgram-nova-3",
        llm="groq-llama-3.1-8b-instant",
        tts="cartesia-sonic-3",
        badge="natural voice",
    ),
    Combo(
        key="deepgram-ollama-elevenlabs",
        label="Deepgram + Ollama + ElevenLabs",
        description="Nova-3 STT · Llama 3.1 8B local · ElevenLabs Turbo. Works without Groq — Ollama must be running.",
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="no groq needed",
    ),
    Combo(
        key="deepgram-gemini-elevenlabs",
        label="Deepgram + Gemini + ElevenLabs",
        description="Nova-3 STT · Gemini 2.0 Flash · ElevenLabs Turbo. Free Gemini key from aistudio.google.com.",
        stt="deepgram-nova-3",
        llm="gemini-gemini-2.0-flash",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="gemini",
    ),
    Combo(
        key="local-oss",
        label="Local OSS (CPU)",
        description="Whisper tiny · Qwen2.5 0.5B (Ollama) · Piper. Fully offline STT/LLM/TTS, real-time on CPU.",
        stt="local-whisper-tiny",
        llm="ollama-qwen2.5:0.5b-instruct",
        tts="piper-en_US-lessac-medium",
        badge="offline",
    ),
]


# Catalog used to populate the precall page's manual-override dropdowns.
STT_OPTIONS = [
    {"key": "groq-whisper-large-v3-turbo", "label": "Groq · Whisper large-v3-turbo  (free, no cap)",
     "group": "Cloud — Groq"},
    {"key": "groq-whisper-large-v3",       "label": "Groq · Whisper large-v3  (max accuracy)",
     "group": "Cloud — Groq"},
    {"key": "deepgram-nova-3",             "label": "Deepgram · Nova-3  (best Indian English)",
     "group": "Cloud — Deepgram"},
    {"key": "deepgram-nova-2",             "label": "Deepgram · Nova-2",
     "group": "Cloud — Deepgram"},
    {"key": "local-whisper-tiny",          "label": "Local · Whisper tiny  (CPU, offline)",
     "group": "Local — OSS"},
    {"key": "local-whisper-base",          "label": "Local · Whisper base  (CPU, slower, more accurate)",
     "group": "Local — OSS"},
]

LLM_OPTIONS = [
    {"key": "groq-llama-3.1-8b-instant",         "label": "Groq · Llama 3.1 · 8B Instant  ★ default",
     "group": "Cloud — Groq"},
    {"key": "groq-llama-3.3-70b-versatile",      "label": "Groq · Llama 3.3 · 70B Versatile (higher quality)",
     "group": "Cloud — Groq"},
    {"key": "groq-openai/gpt-oss-120b",          "label": "Groq · GPT-OSS 120B (top quality, slower)",
     "group": "Cloud — Groq"},
    {"key": "groq-openai/gpt-oss-20b",           "label": "Groq · GPT-OSS 20B",
     "group": "Cloud — Groq"},
    {"key": "groq-meta-llama/llama-4-scout-17b-16e-instruct", "label": "Groq · Llama 4 Scout 17B",
     "group": "Cloud — Groq"},
    {"key": "gemini-gemini-2.0-flash",            "label": "Gemini · 2.0 Flash  (free, fast — needs GEMINI_API_KEY)",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-1.5-flash",            "label": "Gemini · 1.5 Flash  (free, stable — needs GEMINI_API_KEY)",
     "group": "Cloud — Google"},
    {"key": "ollama-llama3.1:8b-instruct-q4_K_M", "label": "Ollama · Llama 3.1 8B (best local quality)",
     "group": "Local — Ollama"},
    {"key": "ollama-llama3.2:3b",                 "label": "Ollama · Llama 3.2 3B (fast, good quality)",
     "group": "Local — Ollama"},
    {"key": "ollama-llama3.2:1b",                 "label": "Ollama · Llama 3.2 1B (fastest local)",
     "group": "Local — Ollama"},
    {"key": "ollama-qwen2.5:0.5b-instruct",       "label": "Ollama · Qwen2.5 0.5B Instruct  (CPU, offline)",
     "group": "Local — Ollama"},
    {"key": "ollama-qwen2.5:1.5b-instruct",       "label": "Ollama · Qwen2.5 1.5B Instruct",
     "group": "Local — Ollama"},
]

TTS_OPTIONS = [
    {"key": "elevenlabs-eleven_turbo_v2_5",   "label": "ElevenLabs · Turbo v2.5  ★ default",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",   "label": "ElevenLabs · Flash v2.5  (lowest latency)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_multilingual_v2", "label": "ElevenLabs · Multilingual v2 (29 languages)",
     "group": "Cloud — ElevenLabs"},
    {"key": "cartesia-sonic-3",               "label": "Cartesia · Sonic-3  (most natural)",
     "group": "Cloud — Cartesia"},
    {"key": "cartesia-sonic-turbo",           "label": "Cartesia · Sonic Turbo (lowest latency)",
     "group": "Cloud — Cartesia"},
    {"key": "piper-en_US-lessac-medium",      "label": "Piper · lessac (neutral US female, CPU)  ★ offline default",
     "group": "Local — OSS"},
    {"key": "piper-en_US-ryan-high",          "label": "Piper · ryan-high (US male, high quality)",
     "group": "Local — OSS"},
    {"key": "piper-en_US-arctic-medium",      "label": "Piper · arctic-medium (US female, alternative)",
     "group": "Local — OSS"},
    {"key": "piper-en_GB-alan-medium",        "label": "Piper · alan-medium (British male)",
     "group": "Local — OSS"},
    {"key": "kokoro-af_bella",                "label": "Kokoro · af_bella  (warm female, ~150 MB download)",
     "group": "Local — OSS"},
    {"key": "kokoro-af_sarah",                "label": "Kokoro · af_sarah  (~150 MB download)",
     "group": "Local — OSS"},
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
