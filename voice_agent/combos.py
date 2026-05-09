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
# Curated 9-combo set, each tuned for ONE specific scenario the operator
# might face. No two combos compete for the same niche — pick by what the
# call is FOR, not by which provider is hot today. Underlying STT / LLM /
# TTS plugins are still all switchable via the precall page's manual
# dropdowns (STT_OPTIONS / LLM_OPTIONS / TTS_OPTIONS) for power users.
#
# Recipe rule of thumb:
#   Live calls need <500ms first-token end-to-end. STT (Deepgram Nova-3)
#   and TTS (Cartesia Sonic-2) are constant across cloud combos because
#   they're each best-in-class for their layer at <200ms. The LLM is
#   what we vary by use case.
PRESETS: list[Combo] = [
    # ═════════════════════════════════════════════════════════════════
    # 🏆 Production / Sales — the default. Most human-feeling on a call.
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="production-sales",
        label="🏆 Production sales (default · most human)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Llama 3.3 70B (~388ms, smart enough to qualify)  →  "
            "TTS: Cartesia Sonic-2 (expressive, <100ms streaming). "
            "Pick this for high-value outbound calls."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="default · most human",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 💰 Volume / Bulk — cheapest LLM, still emotional voice
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="volume-bulk",
        label="💰 Volume / bulk dialing (cheapest, still emotional)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Micro via Bedrock ap-south-1 ($0.03/10k turns)  →  "
            "TTS: Cartesia Sonic-2 (expressive). "
            "Pick this when calling 100s of leads — cheapest LLM per turn."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="cartesia-sonic-2",
        badge="cheapest cloud",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🇮🇳 India-routed — best latency for callers in India
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="india-routed",
        label="🇮🇳 India-routed (Mumbai region, balanced)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Pro via Bedrock ap-south-1 (~848ms, India-region)  →  "
            "TTS: Cartesia Sonic-2 (expressive). "
            "Pick when caller is in India and you want geographic routing."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-pro-v1:0",
        tts="cartesia-sonic-2",
        badge="India region",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🧠 Smart / Qualification — best LLM for complex B2B conversations
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="smart-qualification",
        label="🧠 Smart qualification (Nemotron, NVIDIA-tuned for chat)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Nemotron 70B (NVIDIA-tuned for conversational reasoning)  →  "
            "TTS: Cartesia Sonic-2. "
            "Pick for B2B / enterprise calls where the agent needs to reason across "
            "objections, multi-step asks, technical Q&A."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="cartesia-sonic-2",
        badge="smartest",
    ),

    # ═════════════════════════════════════════════════════════════════
    # ⚡ Fastest cloud — Groq LPUs (EC2-only — blocked on Indian ISPs)
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="fastest-cloud",
        label="⚡ Fastest cloud (Groq LPUs — EC2-only)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Groq Llama 3.3 70B Versatile (~250ms — fastest LLM silicon)  →  "
            "TTS: Cartesia Sonic-2. "
            "Pick this when deployed to a cloud VM. Blocked on Indian residential ISPs."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.3-70b-versatile",
        tts="cartesia-sonic-2",
        badge="EC2-only · fastest",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 💎 Voice cloning — premium, your own voice via ElevenLabs
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="voice-cloning",
        label="💎 Voice cloning (your voice, ElevenLabs paid plan)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Llama 3.3 70B  →  "
            "TTS: ElevenLabs Turbo v2.5 with cloned voice (paid Starter $5/mo). "
            "Pick this for follow-up calls where the customer expects YOUR voice."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="paid · cloned voice",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🔒 Privacy / Local LLM — sensitive data, prompts stay on-prem
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="privacy-local-llm",
        label="🔒 Privacy mode (local LLM, cloud STT/TTS only)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Llama 3.1 8B running on local Ollama (no cloud reasoning)  →  "
            "TTS: Cartesia Sonic-2. "
            "Pick this when prompts may contain sensitive customer data — only "
            "raw audio transits cloud STT/TTS, never the conversation reasoning."
        ),
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="cartesia-sonic-2",
        badge="private LLM",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🏃 Budget low-latency — Bedrock Lite + Cartesia, India route
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="budget-india-fast",
        label="🏃 Budget India fast (Nova Lite + Cartesia)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 ($0.05/10k turns, ~812ms)  →  "
            "TTS: Cartesia Sonic-2 (expressive). "
            "Sweet spot for India ops — faster than Pro, cheap, still emotional."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="cartesia-sonic-2",
        badge="balanced India",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 📦 Offline — air-gapped, zero cloud, lowest quality
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="offline-airgapped",
        label="📦 Offline / air-gapped (zero cloud)",
        description=(
            "STT: faster-whisper tiny on CPU  →  "
            "LLM: Qwen2.5 0.5B on Ollama  →  "
            "TTS: Piper en_US-lessac on CPU. "
            "Pick for demos, air-gapped tests, or when internet is unreliable. "
            "Quality is intentionally minimal — this is the no-cloud floor."
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
