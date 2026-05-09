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
    # Subjective overall pipeline quality. One of:
    #   "bestest"  — top-tier production. Best STT + best LLM + best TTS.
    #   "best"     — strong production pick with one tradeoff (cost / region / EC2).
    #   "average"  — viable but compromised (older STT, cheaper LLM, or flatter TTS).
    quality: str = "average"


# Default = first entry. Picked when no combo/stt/llm/tts is specified.
#
# 10 full-pipeline combos. Each one is a complete STT × LLM × TTS stack
# you can dial test as a unit. No two combos use the same triplet —
# every pipeline tells a coherent design story (premium / India / speed /
# budget / privacy / etc).
#
# Mix coverage across providers:
#   STT  : Deepgram Nova-3 / Nova-2,  Groq Whisper Large-v3,  Local Whisper tiny / base
#   LLM  : NVIDIA H100 (Llama 70B, Nemotron, 8B),  Bedrock Mumbai
#          (Nova Pro / Lite / Micro),  Groq LPU (Llama 70B), Local Ollama
#   TTS  : Cartesia Sonic-2 / Sonic-English,  Deepgram Aura-2,
#          ElevenLabs Turbo / Flash,  Piper offline
#
# This way you can A/B any pipeline by combo without juggling dropdowns.
PRESETS: list[Combo] = [
    # ═══════════════════════════════════════════════════════════════════
    # 🏆 PREMIUM ALL-CLOUD (DEFAULT) — best of each layer
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="premium-allcloud",
        label="🏆 Premium all-cloud (default)",
        description=(
            "STT: Deepgram Nova-3 (best Indian English)  →  "
            "LLM: NVIDIA Llama 3.3 70B (H100, ~388ms)  →  "
            "TTS: Cartesia Sonic-2 (expressive, <100ms streaming). "
            "Highest quality at every layer. The production default."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="default · premium",
        quality="bestest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🇮🇳 ALL-INDIA ROUTED — Mumbai region for the LLM hop
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="india-routed",
        label="🇮🇳 India-routed (Bedrock Mumbai + Cartesia)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Pro via Bedrock ap-south-1 (~848ms, India region)  →  "
            "TTS: Cartesia Sonic-2. Geographic optimisation for Indian callers."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-pro-v1:0",
        tts="cartesia-sonic-2",
        badge="India region",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # ⚡ ALL-GROQ SPEED — fastest end-to-end pipeline (EC2-only)
    # Groq Whisper STT + Groq Llama LLM + ElevenLabs Flash TTS
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="speed-allgroq",
        label="⚡ Speed all-Groq (EC2-only · sub-500ms total)",
        description=(
            "STT: Groq Whisper Large-v3 (~150ms)  →  "
            "LLM: Groq Llama 3.3 70B Versatile (~250ms LPU silicon)  →  "
            "TTS: ElevenLabs Flash v2.5 (fastest premium TTS). "
            "Fastest end-to-end. Blocked from Indian residential ISPs — "
            "works once deployed to a cloud VM."
        ),
        stt="groq-whisper-large-v3",
        llm="groq-llama-3.3-70b-versatile",
        tts="elevenlabs-eleven_flash_v2_5",
        badge="EC2-only · fastest",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 💰 BUDGET PIPELINE — cheapest cloud combo possible
    # Older Deepgram + cheapest Bedrock LLM + Deepgram Aura
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="budget-cloud",
        label="💰 Budget pipeline (cheapest cloud)",
        description=(
            "STT: Deepgram Nova-2 (older, cheaper)  →  "
            "LLM: Amazon Nova Micro Mumbai ($0.03/10k turns)  →  "
            "TTS: Deepgram Aura-2 Asteria (cheapest premium TTS). "
            "Cheapest viable cloud combo per call."
        ),
        stt="deepgram-nova-2",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest",
        quality="average",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🧠 SMART B2B — best reasoning LLM + premium voice
    # Nemotron is NVIDIA's chat-tuned 70B — better at multi-turn reasoning
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="smart-b2b",
        label="🧠 Smart B2B (Nemotron + ElevenLabs Turbo)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA Nemotron 70B (NVIDIA-tuned for conversational reasoning)  →  "
            "TTS: ElevenLabs Turbo v2.5 (premium voice quality). "
            "For high-stakes B2B / enterprise calls needing multi-turn reasoning."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="smartest · paid TTS",
        quality="bestest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🎤 GROQ-STT EXPERIMENT — see if Groq's Whisper beats Deepgram
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="groq-stt-mix",
        label="🎤 Groq STT + NVIDIA + Cartesia (EC2-only)",
        description=(
            "STT: Groq Whisper Large-v3 (~150ms LPU silicon)  →  "
            "LLM: NVIDIA Llama 3.3 70B  →  "
            "TTS: Cartesia Sonic-2. "
            "Tests whether Groq's Whisper STT outperforms Deepgram on US/cloud calls. "
            "EC2-only."
        ),
        stt="groq-whisper-large-v3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="EC2-only · STT test",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🇮🇳 INDIA + AURA — pure-Mumbai LLM with cheaper TTS option
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="india-cheap-aura",
        label="🇮🇳 India cheap + Aura (Nova Lite + Aura Thalia)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 ($0.05/10k turns)  →  "
            "TTS: Deepgram Aura-2 Thalia (warm female voice). "
            "Sweet spot for India ops — cheap LLM, decent voice, all geo-routed."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-thalia-en",
        badge="balanced India",
        quality="average",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🔒 PRIVACY HYBRID — local STT + local LLM + cloud TTS
    # Audio reasoning never leaves the machine; only synthesis is cloud.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="privacy-hybrid",
        label="🔒 Privacy hybrid (local STT + LLM, cloud TTS)",
        description=(
            "STT: faster-whisper base on CPU  →  "
            "LLM: Llama 3.1 8B on local Ollama  →  "
            "TTS: Cartesia Sonic-2 (cloud — only voice synthesis is remote). "
            "Sensitive prompts never reach a cloud LLM. Tradeoff: ~500-800ms STT."
        ),
        stt="local-whisper-base",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="cartesia-sonic-2",
        badge="private LLM",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 💎 VOICE-CLONING PREMIUM — your own voice via ElevenLabs
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="voice-cloning",
        label="💎 Voice cloning (your voice via ElevenLabs)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA Llama 3.3 70B  →  "
            "TTS: ElevenLabs Turbo v2.5 with your cloned voice. "
            "Requires paid ElevenLabs Starter ($5/mo). "
            "Set ELEVENLABS_VOICE_ID in .env to your cloned voice ID."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="paid · cloned voice",
        quality="bestest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 📦 PURE OFFLINE — air-gapped, zero cloud dependencies
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="offline-airgapped",
        label="📦 Pure offline (zero cloud)",
        description=(
            "STT: faster-whisper tiny on CPU  →  "
            "LLM: Qwen2.5 0.5B on Ollama  →  "
            "TTS: Piper en_US-lessac on CPU. "
            "Zero cloud calls — for air-gapped demos / unreliable internet. "
            "Quality is intentionally the no-cloud floor."
        ),
        stt="local-whisper-tiny",
        llm="ollama-qwen2.5:0.5b-instruct",
        tts="piper-en_US-lessac-medium",
        badge="offline",
        quality="average",
    ),
]


# ─── Manual override dropdowns on the precall page ────────────────────
STT_OPTIONS = [
    {"key": "deepgram-nova-3",        "label": "Deepgram · Nova-3  (best Indian English) ★",
     "group": "Cloud — Deepgram"},
    {"key": "deepgram-nova-2",        "label": "Deepgram · Nova-2  (older, cheaper)",
     "group": "Cloud — Deepgram"},
    {"key": "groq-whisper-large-v3",  "label": "Groq · Whisper Large-v3  (~150ms · EC2-only)",
     "group": "Cloud — Groq"},
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
