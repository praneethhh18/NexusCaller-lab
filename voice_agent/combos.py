"""
Preset stack combos for the precall config page.

Each combo is a (label, stt, llm, tts) tuple where the keys are what the
LiveKit Agent's plugin builder understands:

    STT keys: deepgram-nova-3 | deepgram-nova-2 | local-whisper-tiny
    LLM keys: bedrock-<model-id> | gemini-<model> | openai-<model>
              | ollama-<model> | groq-<model>
    TTS keys: deepgram-aura-2-<voice> | elevenlabs-<model>
              | piper-<voice>

Combos are organized into three performance tiers below.
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
    badge: str = ""   # optional pill ("default", "fastest", ...)


# First combo is the default — picked when no combo/stt/llm/tts is specified.
PRESETS: list[Combo] = [
    # ── ⚡ FASTEST — sub-700ms reply, lowest cost ────────────────────────
    Combo(
        key="fast-bedrock-haiku",
        label="⚡ Fast · Claude Haiku 4.5",
        description="Nova-3 STT · Claude Haiku 4.5 (Bedrock) · Aura-2 Asteria. ~600ms reply, $0.001/turn.",
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="default",
    ),
    Combo(
        key="fast-bedrock-novalite",
        label="⚡ Fast · Nova Lite",
        description="Nova-3 STT · Amazon Nova Lite (Bedrock) · Aura-2 Asteria. ~500ms reply, cheapest.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest",
    ),
    Combo(
        key="fast-bedrock-novamicro",
        label="⚡ Ultra-fast · Nova Micro",
        description="Nova-3 STT · Amazon Nova Micro (Bedrock) · Aura-2 Asteria. Tiny model, ~400ms, $0.000035/1k tok.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-micro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="ultra-fast",
    ),

    # ── ⚖️ BALANCED — best quality-per-latency for sales calls ──────────
    Combo(
        key="balanced-bedrock-sonnet",
        label="⚖️ Balanced · Claude Sonnet 4.5",
        description="Nova-3 STT · Claude Sonnet 4.5 (Bedrock) · Aura-2 Asteria. ~900ms reply, sharper reasoning.",
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="recommended",
    ),
    Combo(
        key="balanced-bedrock-novapro",
        label="⚖️ Balanced · Nova Pro",
        description="Nova-3 STT · Amazon Nova Pro (Bedrock) · Aura-2 Asteria. Solid all-rounder.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-pro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="",
    ),

    # ── 🧠 HIGHEST QUALITY — when the conversation has to land ──────────
    Combo(
        key="quality-bedrock-opus",
        label="🧠 Quality · Claude Opus 4",
        description="Nova-3 STT · Claude Opus 4 (Bedrock) · Aura-2 Orion male. Best reasoning, ~1.4s reply.",
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-opus-4-1-20250805-v1:0",
        tts="deepgram-aura-2-orion-en",
        badge="best quality",
    ),

    # ── 🔁 ALTERNATIVES — used when Bedrock keys aren't set ──────────────
    Combo(
        key="alt-gemini-flash",
        label="🔁 Alt · Gemini 2.5 Flash",
        description="Nova-3 STT · Gemini 2.5 Flash · Aura-2 Asteria. Free Google API, no AWS needed.",
        stt="deepgram-nova-3",
        llm="gemini-gemini-2.5-flash",
        tts="deepgram-aura-2-asteria-en",
        badge="free",
    ),
    Combo(
        key="alt-ollama-local",
        label="🔁 Alt · Ollama (local LLM)",
        description="Nova-3 STT · Llama 3.1 8B local Ollama · Aura-2. Zero cloud-LLM cost (Ollama must be running).",
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="deepgram-aura-2-asteria-en",
        badge="local LLM",
    ),

    # ── 📦 OFFLINE — fully local, no cloud at all ────────────────────────
    Combo(
        key="offline-local",
        label="📦 Offline · all-local stack",
        description="Whisper tiny · Qwen2.5 0.5B (Ollama) · Piper TTS. Zero cloud calls, real-time CPU.",
        stt="local-whisper-tiny",
        llm="ollama-qwen2.5:0.5b-instruct",
        tts="piper-en_US-lessac-medium",
        badge="offline",
    ),
]


STT_OPTIONS = [
    {"key": "deepgram-nova-3",        "label": "Deepgram · Nova-3  (best Indian English) ★",
     "group": "Cloud — Deepgram"},
    {"key": "deepgram-nova-2",        "label": "Deepgram · Nova-2",
     "group": "Cloud — Deepgram"},
    {"key": "local-whisper-tiny",     "label": "Local · Whisper tiny  (CPU, offline)",
     "group": "Local — Whisper"},
    {"key": "local-whisper-base",     "label": "Local · Whisper base  (slower, more accurate)",
     "group": "Local — Whisper"},
]

LLM_OPTIONS = [
    # Bedrock — recommended primary
    {"key": "bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                          "label": "Bedrock · Claude Haiku 4.5  ★ default",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                                          "label": "Bedrock · Claude Sonnet 4.5",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.anthropic.claude-opus-4-1-20250805-v1:0",
                                          "label": "Bedrock · Claude Opus 4",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.amazon.nova-pro-v1:0",
                                          "label": "Bedrock · Amazon Nova Pro",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.amazon.nova-lite-v1:0",
                                          "label": "Bedrock · Amazon Nova Lite  (cheapest)",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.amazon.nova-micro-v1:0",
                                          "label": "Bedrock · Amazon Nova Micro  (ultra-fast)",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.meta.llama3-3-70b-instruct-v1:0",
                                          "label": "Bedrock · Meta Llama 3.3 70B",
     "group": "Cloud — AWS Bedrock"},
    # Gemini — free fallback
    {"key": "gemini-gemini-2.5-flash",    "label": "Gemini · 2.5 Flash  (free)",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-2.5-flash-lite", "label": "Gemini · 2.5 Flash Lite",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-2.5-pro",      "label": "Gemini · 2.5 Pro  (best quality)",
     "group": "Cloud — Google"},
    # OpenAI
    {"key": "openai-gpt-4o-mini",         "label": "OpenAI · GPT-4o mini",
     "group": "Cloud — OpenAI"},
    {"key": "openai-gpt-4o",              "label": "OpenAI · GPT-4o",
     "group": "Cloud — OpenAI"},
    # Groq (works only on networks where Groq's CDN isn't blocked)
    {"key": "groq-llama-3.1-8b-instant",  "label": "Groq · Llama 3.1 8B  (CDN-blocked on some ISPs)",
     "group": "Cloud — Groq"},
    # Local Ollama
    {"key": "ollama-llama3.1:8b-instruct-q4_K_M",
                                          "label": "Ollama · Llama 3.1 8B  (local)",
     "group": "Local — Ollama"},
    {"key": "ollama-llama3.2:3b",         "label": "Ollama · Llama 3.2 3B  (fast local)",
     "group": "Local — Ollama"},
    {"key": "ollama-qwen2.5:0.5b-instruct", "label": "Ollama · Qwen2.5 0.5B  (tiny local)",
     "group": "Local — Ollama"},
]

TTS_OPTIONS = [
    {"key": "deepgram-aura-2-asteria-en", "label": "Deepgram · Aura-2 Asteria  (US female) ★",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-thalia-en",  "label": "Deepgram · Aura-2 Thalia  (US female warm)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-luna-en",    "label": "Deepgram · Aura-2 Luna  (US female soft)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-orion-en",   "label": "Deepgram · Aura-2 Orion  (US male)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-asteria-en",   "label": "Deepgram · Aura v1 Asteria  (legacy)",
     "group": "Cloud — Deepgram Aura"},
    # ElevenLabs (locked on free tier — needs paid plan)
    {"key": "elevenlabs-eleven_turbo_v2_5",   "label": "ElevenLabs · Turbo v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",   "label": "ElevenLabs · Flash v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},
    # Local Piper
    {"key": "piper-en_US-lessac-medium",  "label": "Piper · lessac  (US female, offline)",
     "group": "Local — Piper"},
    {"key": "piper-en_US-ryan-high",      "label": "Piper · ryan-high  (US male, offline)",
     "group": "Local — Piper"},
    {"key": "piper-en_GB-alan-medium",    "label": "Piper · alan-medium  (British male, offline)",
     "group": "Local — Piper"},
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
