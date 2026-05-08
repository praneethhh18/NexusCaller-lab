"""
Preset stack combos for the precall config page.

Each combo is a (label, stt, llm, tts) tuple where the keys are what the
LiveKit Agent's plugin builder understands.

Combos are organized by performance tier (fastest → quality → alts).
The first entry is the default. All Bedrock models are referenced via
their inference-profile id (us.<...>) which doesn't require a use-case
form and works on-demand.
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


PRESETS: list[Combo] = [
    # ─── ⚡ FASTEST + CHEAPEST — best for high-volume sales outbound ────
    Combo(
        key="fast-nova-micro",
        label="⚡ Nova Micro",
        description="Nova-3 STT · Amazon Nova Micro (Bedrock) · Aura-2. ~150ms first token, $0.035/1M in. Cheapest cloud LLM that still holds a conversation.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-micro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="default",
    ),
    Combo(
        key="fast-llama-3b",
        label="⚡ Llama 3.2 3B",
        description="Nova-3 STT · Meta Llama 3.2 3B (Bedrock) · Aura-2. Open-weights small model, $0.15/1M, ~250ms first token. Sweet-spot for cost/quality.",
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-2-3b-instruct-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="open-source",
    ),
    Combo(
        key="fast-nova-lite",
        label="⚡ Nova Lite",
        description="Nova-3 STT · Amazon Nova Lite (Bedrock) · Aura-2. Slight quality bump over Micro, $0.06/1M in.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheap",
    ),

    # ─── ⚖️ BALANCED — when Nova Micro feels too short ──────────────────
    Combo(
        key="balanced-llama-8b",
        label="⚖️ Llama 3.1 8B",
        description="Nova-3 STT · Meta Llama 3.1 8B (Bedrock) · Aura-2. Solid open-source, $0.22/1M, ~300ms. Best balance for B2B sales calls.",
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-1-8b-instruct-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="recommended",
    ),
    Combo(
        key="balanced-mistral-7b",
        label="⚖️ Mistral 7B",
        description="Nova-3 STT · Mistral 7B Instruct (Bedrock) · Aura-2. Classic 7B, $0.15/1M, decent reasoning.",
        stt="deepgram-nova-3",
        llm="bedrock-mistral.mistral-7b-instruct-v0:2",
        tts="deepgram-aura-2-asteria-en",
        badge="",
    ),
    Combo(
        key="balanced-mixtral",
        label="⚖️ Mixtral 8x7B (MoE)",
        description="Nova-3 STT · Mistral Mixtral 8x7B (Bedrock) · Aura-2. Mixture-of-experts, fast for its capability.",
        stt="deepgram-nova-3",
        llm="bedrock-mistral.mixtral-8x7b-instruct-v0:1",
        tts="deepgram-aura-2-asteria-en",
        badge="",
    ),
    Combo(
        key="balanced-nova-pro",
        label="⚖️ Nova Pro",
        description="Nova-3 STT · Amazon Nova Pro (Bedrock) · Aura-2. AWS's mid-tier, $0.80/1M.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-pro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="",
    ),

    # ─── 🧠 QUALITY — high-stakes calls ─────────────────────────────────
    Combo(
        key="quality-llama-70b",
        label="🧠 Llama 3.3 70B",
        description="Nova-3 STT · Meta Llama 3.3 70B (Bedrock) · Aura-2 Orion. Best open-source 70B, $0.72/1M.",
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-3-70b-instruct-v1:0",
        tts="deepgram-aura-2-orion-en",
        badge="best open",
    ),
    Combo(
        key="quality-claude-sonnet",
        label="🧠 Claude Sonnet 4.5",
        description="Nova-3 STT · Claude Sonnet 4.5 (Bedrock) · Aura-2 Orion. Best general reasoning, $3.00/1M (use sparingly).",
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tts="deepgram-aura-2-orion-en",
        badge="premium",
    ),

    # ─── 🔁 ALTERNATIVES — when Bedrock isn't configured ────────────────
    Combo(
        key="alt-gemini-flash",
        label="🔁 Gemini 2.5 Flash (free)",
        description="Nova-3 STT · Gemini 2.5 Flash (Google AI Studio) · Aura-2. No AWS needed, free quota.",
        stt="deepgram-nova-3",
        llm="gemini-gemini-2.5-flash",
        tts="deepgram-aura-2-asteria-en",
        badge="free",
    ),
    Combo(
        key="alt-ollama",
        label="🔁 Ollama (local Llama 3.1 8B)",
        description="Nova-3 STT · Llama 3.1 8B local · Aura-2. Zero LLM cost (Ollama running locally).",
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="deepgram-aura-2-asteria-en",
        badge="local LLM",
    ),

    # ─── 📦 OFFLINE — fully local, no cloud at all ──────────────────────
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
    # Bedrock — fastest/cheapest first
    {"key": "bedrock-us.amazon.nova-micro-v1:0",
                                          "label": "Bedrock · Nova Micro  ★ default ($0.035/1M)",
     "group": "Cloud — AWS Bedrock · Fast"},
    {"key": "bedrock-us.meta.llama3-2-3b-instruct-v1:0",
                                          "label": "Bedrock · Llama 3.2 3B  ($0.15/1M)",
     "group": "Cloud — AWS Bedrock · Fast"},
    {"key": "bedrock-us.amazon.nova-lite-v1:0",
                                          "label": "Bedrock · Nova Lite  ($0.06/1M)",
     "group": "Cloud — AWS Bedrock · Fast"},
    # Bedrock — balanced
    {"key": "bedrock-us.meta.llama3-1-8b-instruct-v1:0",
                                          "label": "Bedrock · Llama 3.1 8B  ($0.22/1M)",
     "group": "Cloud — AWS Bedrock · Balanced"},
    {"key": "bedrock-mistral.mistral-7b-instruct-v0:2",
                                          "label": "Bedrock · Mistral 7B  ($0.15/1M)",
     "group": "Cloud — AWS Bedrock · Balanced"},
    {"key": "bedrock-mistral.mixtral-8x7b-instruct-v0:1",
                                          "label": "Bedrock · Mixtral 8x7B MoE  ($0.45/1M)",
     "group": "Cloud — AWS Bedrock · Balanced"},
    {"key": "bedrock-us.amazon.nova-pro-v1:0",
                                          "label": "Bedrock · Nova Pro  ($0.80/1M)",
     "group": "Cloud — AWS Bedrock · Balanced"},
    # Bedrock — quality
    {"key": "bedrock-us.meta.llama3-3-70b-instruct-v1:0",
                                          "label": "Bedrock · Llama 3.3 70B  ($0.72/1M)",
     "group": "Cloud — AWS Bedrock · Quality"},
    {"key": "bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                                          "label": "Bedrock · Claude Sonnet 4.5  ($3.00/1M)",
     "group": "Cloud — AWS Bedrock · Quality"},
    {"key": "bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                          "label": "Bedrock · Claude Haiku 4.5  ($0.80/1M, overpriced for voice)",
     "group": "Cloud — AWS Bedrock · Quality"},
    {"key": "bedrock-mistral.mistral-large-2402-v1:0",
                                          "label": "Bedrock · Mistral Large  ($4.00/1M)",
     "group": "Cloud — AWS Bedrock · Quality"},
    # Gemini — free fallback
    {"key": "gemini-gemini-2.5-flash",    "label": "Gemini · 2.5 Flash  (free, fast)",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-2.5-flash-lite", "label": "Gemini · 2.5 Flash Lite  (free, cheaper)",
     "group": "Cloud — Google"},
    {"key": "gemini-gemini-2.5-pro",      "label": "Gemini · 2.5 Pro  (free quality)",
     "group": "Cloud — Google"},
    # OpenAI
    {"key": "openai-gpt-4o-mini",         "label": "OpenAI · GPT-4o mini",
     "group": "Cloud — OpenAI"},
    {"key": "openai-gpt-4o",              "label": "OpenAI · GPT-4o",
     "group": "Cloud — OpenAI"},
    # Groq (works on networks where Groq's CDN isn't blocked)
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
    # ElevenLabs (free tier banned on this account — needs paid plan)
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
