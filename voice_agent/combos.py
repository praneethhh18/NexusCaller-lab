"""
Preset stack combos for the precall config page.

Each combo is a (label, stt, llm, tts) tuple where the keys are what the
LiveKit Agent's plugin builder understands:

    STT keys: deepgram-nova-3 | deepgram-nova-2 | groq-whisper-large-v3-turbo
    LLM keys: groq-llama-3.1-8b-instant | groq-llama-3.3-70b-versatile
              | groq-openai/gpt-oss-120b
    TTS keys: elevenlabs-eleven_turbo_v2_5 | elevenlabs-eleven_flash_v2_5
              | cartesia-sonic-3 | cartesia-sonic-turbo
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
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
