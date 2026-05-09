"""
Preset stack combos for the precall config page.

Combos are grouped by LLM provider so each provider's combos can be
tested in isolation. Only providers and models with proven low-latency
on this network are included — anything that benchmarked above 1000ms
first-token has been removed.

Pipeline format for every combo:
    STT (speech → text)  →  LLM (text → text)  →  TTS (text → speech)

Live numbers (avg first-token from Indian residential ISP):
  NVIDIA NIM US H100         ~388ms — fastest
  AWS Bedrock Mumbai (apac.) ~812ms — geographically closest
  Groq US                    blocked (Cloudflare ASN) — works on EC2
  Local Ollama / offline     varies with hardware
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Combo:
    key: str          # internal id, sent to the agent in job metadata
    label: str        # human label shown in the picker
    description: str  # one-line caption summarizing the pipeline
    stt: str          # plugin key for STT
    llm: str          # plugin key for LLM
    tts: str          # plugin key for TTS
    badge: str = ""   # optional pill ("default", "cheapest", ...)


# Default = first entry. Picked when no combo/stt/llm/tts is specified.
#
# Why Cartesia Sonic is the new default:
#   - Deepgram Aura is fast but emotionally flat (designed for IVR)
#   - Cartesia Sonic-2 streams in <100ms, has expressive prosody, sounds
#     like a real human on the phone
#   - 30K chars/mo free tier; $5/mo for 100K after
#
# The previous NVIDIA + Aura combos are kept below for users who want
# pure speed at the cost of expressiveness.
PRESETS: list[Combo] = [
    # ═════════════════════════════════════════════════════════════════
    # 💛 Cartesia Sonic — emotional + fast, the new default
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="nvidia-llama70b-cartesia",
        label="💛 NVIDIA + Cartesia Sonic (most human)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Llama 3.3 70B (H100, ~388ms)  →  "
            "TTS: Cartesia Sonic-2 (expressive, <100ms streaming)"
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="default · most human",
    ),
    Combo(
        key="bedrock-novapro-cartesia",
        label="🇮🇳 Bedrock + Cartesia (emotional, India-routed)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Pro via Bedrock ap-south-1  →  "
            "TTS: Cartesia Sonic-2 (expressive, <100ms)"
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-pro-v1:0",
        tts="cartesia-sonic-2",
        badge="emotional · India",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🚀 NVIDIA NIM — H100-hosted, fastest from this network
    # All combos pair with Cartesia Sonic-2 for emotional voice
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="nvidia-nemotron-cartesia",
        label="🚀 NVIDIA · Nemotron 70B (tuned for chat)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Nemotron 70B (NVIDIA-tuned for chat)  →  "
            "TTS: Cartesia Sonic-2"
        ),
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="cartesia-sonic-2",
        badge="NVIDIA-tuned",
    ),
    Combo(
        key="nvidia-llama8b-cartesia",
        label="🚀 NVIDIA · Llama 3.1 8B (cheaper)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Llama 3.1 8B  →  "
            "TTS: Cartesia Sonic-2"
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.1-8b-instruct",
        tts="cartesia-sonic-2",
        badge="cheaper NIM",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🇮🇳 AWS Bedrock Mumbai (ap-south-1) — geographically closest
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="bedrock-novalite-cartesia",
        label="🇮🇳 Bedrock Mumbai · Nova Lite (cheapest cloud)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 (~812ms)  →  "
            "TTS: Cartesia Sonic-2. $0.05/10k turns."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="cartesia-sonic-2",
        badge="cheap fast",
    ),
    Combo(
        key="bedrock-novamicro-cartesia",
        label="🇮🇳 Bedrock Mumbai · Nova Micro",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Micro via Bedrock ap-south-1  →  "
            "TTS: Cartesia Sonic-2. $0.03/10k turns — cheapest LLM."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="cartesia-sonic-2",
        badge="cheapest",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🌐 Groq — fastest LLM in the world (~150ms) BUT blocked from
    # Indian residential ISPs by Cloudflare ASN. Will work once the
    # worker is deployed to AWS / any non-blocked network.
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="groq-llama70b-cartesia",
        label="🌐 Groq · Llama 3.3 70B  (EC2-only)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Groq Llama 3.3 70B Versatile (~250ms — LPU silicon)  →  "
            "TTS: Cartesia Sonic-2. "
            "Blocked from Indian ISPs; works once deployed to a cloud VM."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.3-70b-versatile",
        tts="cartesia-sonic-2",
        badge="EC2-only",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 📦 Local — runs on your machine, no cloud LLM calls
    # Local-offline kept for users who want zero cloud dependencies;
    # uses Piper (offline TTS) since Cartesia needs internet.
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="local-ollama-cartesia",
        label="📦 Local LLM + Cartesia voice",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Llama 3.1 8B running locally on Ollama  →  "
            "TTS: Cartesia Sonic-2. Zero LLM cost, latency = your CPU."
        ),
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="cartesia-sonic-2",
        badge="local LLM",
    ),
    Combo(
        key="local-offline",
        label="📦 Offline · all-local stack (no internet needed)",
        description=(
            "STT: faster-whisper tiny on CPU  →  "
            "LLM: Qwen2.5 0.5B on Ollama  →  "
            "TTS: Piper en_US-lessac on CPU. Zero cloud calls."
        ),
        stt="local-whisper-tiny",
        llm="ollama-qwen2.5:0.5b-instruct",
        tts="piper-en_US-lessac-medium",
        badge="offline",
    ),
]


# ─── Manual override dropdowns on the precall page ────────────────────
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
    # NVIDIA NIM
    {"key": "nvidia-meta/llama-3.3-70b-instruct",
                                          "label": "Llama 3.3 70B  · ~388ms ★ default",
     "group": "🚀 NVIDIA NIM (US H100)"},
    {"key": "nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
                                          "label": "Nemotron 70B (NVIDIA-tuned)",
     "group": "🚀 NVIDIA NIM (US H100)"},
    {"key": "nvidia-meta/llama-3.1-8b-instruct",
                                          "label": "Llama 3.1 8B",
     "group": "🚀 NVIDIA NIM (US H100)"},
    {"key": "nvidia-mistralai/mistral-7b-instruct-v0.3",
                                          "label": "Mistral 7B v0.3",
     "group": "🚀 NVIDIA NIM (US H100)"},
    # Bedrock Mumbai
    {"key": "bedrock-apac.amazon.nova-lite-v1:0",
                                          "label": "Nova Lite  · ~812ms · $0.05/10k",
     "group": "🇮🇳 Bedrock Mumbai (ap-south-1)"},
    {"key": "bedrock-apac.amazon.nova-pro-v1:0",
                                          "label": "Nova Pro  · ~848ms · $0.76/10k",
     "group": "🇮🇳 Bedrock Mumbai (ap-south-1)"},
    {"key": "bedrock-apac.amazon.nova-micro-v1:0",
                                          "label": "Nova Micro  · ~848ms · $0.03/10k cheapest",
     "group": "🇮🇳 Bedrock Mumbai (ap-south-1)"},
    # Groq (EC2-only)
    {"key": "groq-llama-3.1-8b-instant",  "label": "Llama 3.1 8B Instant  · ~150ms (EC2-only)",
     "group": "🌐 Groq (EC2-only — blocked from Indian ISPs)"},
    {"key": "groq-llama-3.3-70b-versatile", "label": "Llama 3.3 70B Versatile  · ~250ms (EC2-only)",
     "group": "🌐 Groq (EC2-only — blocked from Indian ISPs)"},
    # Local Ollama
    {"key": "ollama-llama3.1:8b-instruct-q4_K_M",
                                          "label": "Llama 3.1 8B  (8GB+ RAM)",
     "group": "📦 Local Ollama"},
    {"key": "ollama-llama3.2:3b",         "label": "Llama 3.2 3B  (faster local)",
     "group": "📦 Local Ollama"},
    {"key": "ollama-qwen2.5:0.5b-instruct", "label": "Qwen2.5 0.5B  (tiny, fits anywhere)",
     "group": "📦 Local Ollama"},
]

TTS_OPTIONS = [
    # Cartesia first — emotionally expressive, our default and primary recommendation
    {"key": "cartesia-sonic-2",            "label": "Sonic-2  (expressive, <100ms streaming) ★",
     "group": "Cloud — Cartesia"},
    {"key": "cartesia-sonic-english",      "label": "Sonic English  (older, also fast)",
     "group": "Cloud — Cartesia"},

    # ElevenLabs kept as a paid alternative for users who want voice-cloning
    {"key": "elevenlabs-eleven_turbo_v2_5",   "label": "Turbo v2.5  (paid plan, voice cloning)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",   "label": "Flash v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},

    # Local / offline only — Piper is the only zero-cloud TTS
    {"key": "piper-en_US-lessac-medium",  "label": "Piper lessac  (offline only)",
     "group": "Local — Piper"},
    {"key": "piper-en_US-ryan-high",      "label": "Piper ryan-high  (offline only)",
     "group": "Local — Piper"},
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
