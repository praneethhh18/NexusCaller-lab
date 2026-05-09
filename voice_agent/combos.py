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
# Test-matrix combo set — organised so each tier holds two layers constant
# and varies the third. This lets the operator A/B compare any one layer
# (STT, LLM, or TTS) without confounding variables. Tiers:
#
#   TIER 1 — production winners (3)         pick by use case
#   TIER 2 — LLM A/B tests (8)              same STT + TTS, swap LLM
#   TIER 3 — TTS A/B tests (4)              same STT + LLM, swap TTS
#   TIER 4 — STT A/B tests (1)              same LLM + TTS, swap STT
#   TIER 5 — Specialty (2)                  voice cloning · offline
#
# Why STT + TTS are held constant in LLM tests:
#   Deepgram Nova-3 (STT) + Cartesia Sonic-2 (TTS) are each best-in-class
#   at <200ms first-byte. The LLM is the layer with the biggest quality
#   variance — that's what most A/B tests should isolate.
PRESETS: list[Combo] = [
    # ═══════════════════════════════════════════════════════════════════
    # 🏆 TIER 1 — PRODUCTION WINNERS — pick by use case
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="best-default",
        label="🏆 Best default (Llama 70B + Cartesia)",
        description=(
            "STT: Deepgram Nova-3  →  LLM: NVIDIA Llama 3.3 70B (~388ms)  →  "
            "TTS: Cartesia Sonic-2 (expressive). The proven all-rounder."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="default",
    ),
    Combo(
        key="best-india",
        label="🇮🇳 Best for India (Bedrock Mumbai)",
        description=(
            "STT: Deepgram Nova-3  →  LLM: Amazon Nova Pro via Bedrock ap-south-1  →  "
            "TTS: Cartesia Sonic-2. Geo-routed for Indian callers."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-pro-v1:0",
        tts="cartesia-sonic-2",
        badge="India region",
    ),
    Combo(
        key="best-budget",
        label="💰 Best budget (Nova Micro · cheapest LLM)",
        description=(
            "STT: Deepgram Nova-3  →  LLM: Amazon Nova Micro Mumbai ($0.03/10k turns)  →  "
            "TTS: Cartesia Sonic-2. Cheapest LLM, still emotional voice."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="cartesia-sonic-2",
        badge="cheapest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🧠 TIER 2 — LLM A/B TESTS — same STT + TTS, swap LLM
    # All hold: STT = Deepgram Nova-3, TTS = Cartesia Sonic-2
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="llm-nvidia-nemotron",
        label="🧠 LLM test · NVIDIA Nemotron 70B (chat-tuned)",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: NVIDIA NIM Nemotron 70B — NVIDIA-tuned for conversational reasoning."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="cartesia-sonic-2",
        badge="LLM test",
    ),
    Combo(
        key="llm-nvidia-llama8b",
        label="🧠 LLM test · NVIDIA Llama 3.1 8B (cheap NIM)",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: NVIDIA NIM Llama 3.1 8B — same H100 infra as 70B but smaller / cheaper."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.1-8b-instruct",
        tts="cartesia-sonic-2",
        badge="LLM test",
    ),
    Combo(
        key="llm-bedrock-novalite",
        label="🧠 LLM test · Bedrock Nova Lite (Mumbai)",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 — $0.05/10k turns."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="cartesia-sonic-2",
        badge="LLM test",
    ),
    Combo(
        key="llm-groq-llama70b",
        label="🧠 LLM test · Groq Llama 3.3 70B (~250ms · EC2-only)",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: Groq Llama 3.3 70B Versatile — fastest LPU silicon. "
            "Blocked from Indian residential ISPs; works once deployed to a cloud VM."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.3-70b-versatile",
        tts="cartesia-sonic-2",
        badge="LLM test · EC2",
    ),
    Combo(
        key="llm-groq-llama8b",
        label="🧠 LLM test · Groq Llama 3.1 8B (~150ms · EC2-only)",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: Groq Llama 3.1 8B Instant — even faster than 70B. "
            "Blocked from Indian residential ISPs."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.1-8b-instant",
        tts="cartesia-sonic-2",
        badge="LLM test · EC2",
    ),
    Combo(
        key="llm-ollama-llama8b",
        label="🧠 LLM test · Local Ollama Llama 3.1 8B",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: Llama 3.1 8B running on local Ollama — no cloud reasoning, "
            "good for sensitive prompts."
        ),
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="cartesia-sonic-2",
        badge="LLM test · local",
    ),
    Combo(
        key="llm-ollama-llama3b",
        label="🧠 LLM test · Local Ollama Llama 3.2 3B",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: Llama 3.2 3B on Ollama — smaller, faster on CPU than 8B."
        ),
        stt="deepgram-nova-3",
        llm="ollama-llama3.2:3b",
        tts="cartesia-sonic-2",
        badge="LLM test · local",
    ),
    Combo(
        key="llm-bedrock-novamicro",
        label="🧠 LLM test · Bedrock Nova Micro (cheapest)",
        description=(
            "Holding STT = Deepgram, TTS = Cartesia. "
            "LLM: Amazon Nova Micro via Bedrock ap-south-1 — $0.03/10k turns, cheapest."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="cartesia-sonic-2",
        badge="LLM test",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🔊 TIER 3 — TTS A/B TESTS — same STT + LLM, swap TTS
    # All hold: STT = Deepgram Nova-3, LLM = NVIDIA Llama 3.3 70B
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="tts-cartesia-english",
        label="🔊 TTS test · Cartesia Sonic English (older)",
        description=(
            "Holding STT = Deepgram, LLM = NVIDIA Llama 70B. "
            "TTS: Cartesia Sonic English — older Sonic model, also fast/expressive."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-english",
        badge="TTS test",
    ),
    Combo(
        key="tts-aura-asteria",
        label="🔊 TTS test · Deepgram Aura Asteria (US female)",
        description=(
            "Holding STT = Deepgram, LLM = NVIDIA Llama 70B. "
            "TTS: Deepgram Aura-2 Asteria — fast US female voice, flatter than Cartesia."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="TTS test",
    ),
    Combo(
        key="tts-aura-orion",
        label="🔊 TTS test · Deepgram Aura Orion (US male)",
        description=(
            "Holding STT = Deepgram, LLM = NVIDIA Llama 70B. "
            "TTS: Deepgram Aura-2 Orion — male voice option."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="deepgram-aura-2-orion-en",
        badge="TTS test",
    ),
    Combo(
        key="tts-elevenlabs-turbo",
        label="🔊 TTS test · ElevenLabs Turbo v2.5 (paid)",
        description=(
            "Holding STT = Deepgram, LLM = NVIDIA Llama 70B. "
            "TTS: ElevenLabs Turbo v2.5 — premium quality, supports voice cloning. "
            "Requires paid ElevenLabs plan."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="TTS test · paid",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🎤 TIER 4 — STT A/B TEST — same LLM + TTS, swap STT
    # All hold: LLM = NVIDIA Llama 3.3 70B, TTS = Cartesia Sonic-2
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="stt-whisper-base",
        label="🎤 STT test · Local Whisper base (offline STT)",
        description=(
            "Holding LLM = NVIDIA Llama 70B, TTS = Cartesia. "
            "STT: faster-whisper base on CPU — no cloud STT, ~500ms on modern CPU. "
            "Tradeoff: slower than Deepgram, but audio never leaves your machine."
        ),
        stt="local-whisper-base",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="STT test · local",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🎯 TIER 5 — SPECIALTY
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="voice-cloning",
        label="💎 Voice cloning (your voice via ElevenLabs)",
        description=(
            "STT: Deepgram Nova-3  →  LLM: NVIDIA Llama 3.3 70B  →  "
            "TTS: ElevenLabs Turbo v2.5 with cloned voice. "
            "Requires paid ElevenLabs Starter ($5/mo). "
            "Set ELEVENLABS_VOICE_ID in .env to your cloned voice ID."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="paid · cloned voice",
    ),
    Combo(
        key="offline-airgapped",
        label="📦 Offline / air-gapped (zero cloud)",
        description=(
            "STT: faster-whisper tiny  →  LLM: Qwen2.5 0.5B on Ollama  →  "
            "TTS: Piper en_US-lessac. Quality is intentionally the no-cloud floor."
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
    # Cartesia first — emotionally expressive, our new default
    {"key": "cartesia-sonic-2",            "label": "Sonic-2  (expressive, <100ms streaming) ★",
     "group": "Cloud — Cartesia"},
    {"key": "cartesia-sonic-english",      "label": "Sonic English  (older, also fast)",
     "group": "Cloud — Cartesia"},

    {"key": "deepgram-aura-2-asteria-en", "label": "Aura-2 Asteria  (US female, flat)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-thalia-en",  "label": "Aura-2 Thalia  (US female warm)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-luna-en",    "label": "Aura-2 Luna  (US female soft)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-orion-en",   "label": "Aura-2 Orion  (US male)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "elevenlabs-eleven_turbo_v2_5",   "label": "Turbo v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",   "label": "Flash v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},
    {"key": "piper-en_US-lessac-medium",  "label": "Piper lessac  (US female, offline)",
     "group": "Local — Piper"},
    {"key": "piper-en_US-ryan-high",      "label": "Piper ryan-high  (US male, offline)",
     "group": "Local — Piper"},
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
