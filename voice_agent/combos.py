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
PRESETS: list[Combo] = [
    # ═════════════════════════════════════════════════════════════════
    # 🚀 NVIDIA NIM — H100-hosted, fastest from this network (~388ms)
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="nvidia-llama-70b",
        label="🚀 NVIDIA · Llama 3.3 70B",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Llama 3.3 70B (H100, ~388ms)  →  "
            "TTS: Deepgram Aura-2 Asteria"
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="default · fastest",
    ),
    Combo(
        key="nvidia-nemotron-70b",
        label="🚀 NVIDIA · Nemotron 70B (tuned)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Nemotron 70B (NVIDIA-tuned for chat)  →  "
            "TTS: Deepgram Aura-2 Asteria"
        ),
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="NVIDIA-tuned",
    ),
    Combo(
        key="nvidia-llama-8b",
        label="🚀 NVIDIA · Llama 3.1 8B",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA NIM Llama 3.1 8B  →  "
            "TTS: Deepgram Aura-2 Asteria"
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.1-8b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="cheaper NIM",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🇮🇳 AWS Bedrock Mumbai (ap-south-1) — geographically closest
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="bedrock-nova-lite",
        label="🇮🇳 Bedrock Mumbai · Nova Lite",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 (~812ms)  →  "
            "TTS: Deepgram Aura-2 Asteria. $0.05/10k turns."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheap fast",
    ),
    Combo(
        key="bedrock-nova-pro",
        label="🇮🇳 Bedrock Mumbai · Nova Pro",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Pro via Bedrock ap-south-1 (~848ms)  →  "
            "TTS: Deepgram Aura-2 Asteria. Better reasoning than Lite."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-pro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="balanced",
    ),
    Combo(
        key="bedrock-nova-micro",
        label="🇮🇳 Bedrock Mumbai · Nova Micro",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Micro via Bedrock ap-south-1 (~848ms)  →  "
            "TTS: Deepgram Aura-2 Asteria. $0.03/10k turns — cheapest."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest",
    ),
    Combo(
        key="aws-mumbai-full",
        label="🇮🇳 All-AWS Mumbai · Nova Lite + Polly Kajal",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Lite (Bedrock ap-south-1)  →  "
            "TTS: Amazon Polly Kajal (Indian female, Mumbai). "
            "LLM↔TTS stays within AWS Mumbai — needs Polly IAM permission."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="polly-Kajal-generative",
        badge="all-AWS",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 🌐 Groq — fastest LLM in the world (~150ms) BUT blocked from
    # Indian residential ISPs by Cloudflare ASN. Will work once the
    # worker is deployed to AWS / any non-blocked network.
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="groq-llama-8b",
        label="🌐 Groq · Llama 3.1 8B Instant  (EC2-only)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Groq Llama 3.1 8B (~150ms — LPU silicon)  →  "
            "TTS: Deepgram Aura-2 Asteria. "
            "Blocked from Indian ISPs; works once deployed to a cloud VM."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.1-8b-instant",
        tts="deepgram-aura-2-asteria-en",
        badge="fastest if unblocked",
    ),
    Combo(
        key="groq-llama-70b",
        label="🌐 Groq · Llama 3.3 70B Versatile  (EC2-only)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Groq Llama 3.3 70B Versatile (~250ms)  →  "
            "TTS: Deepgram Aura-2 Asteria. "
            "Blocked from Indian ISPs; works once deployed to a cloud VM."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.3-70b-versatile",
        tts="deepgram-aura-2-asteria-en",
        badge="EC2-only",
    ),

    # ═════════════════════════════════════════════════════════════════
    # 📦 Local — runs on your machine, no cloud LLM calls
    # ═════════════════════════════════════════════════════════════════
    Combo(
        key="local-ollama",
        label="📦 Local · Ollama Llama 3.1 8B",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Llama 3.1 8B running locally on Ollama  →  "
            "TTS: Deepgram Aura-2 Asteria. Zero LLM cost, latency = your CPU."
        ),
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="deepgram-aura-2-asteria-en",
        badge="local LLM",
    ),
    Combo(
        key="local-offline",
        label="📦 Offline · all-local stack",
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
    {"key": "deepgram-aura-2-asteria-en", "label": "Aura-2 Asteria  (US female) ★",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-thalia-en",  "label": "Aura-2 Thalia  (US female warm)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-luna-en",    "label": "Aura-2 Luna  (US female soft)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "deepgram-aura-2-orion-en",   "label": "Aura-2 Orion  (US male)",
     "group": "Cloud — Deepgram Aura"},
    {"key": "polly-Kajal-generative",     "label": "Polly · Kajal generative  (Indian female, en-IN)",
     "group": "Cloud — Amazon Polly (Mumbai, needs IAM perm)"},
    {"key": "polly-Kajal-neural",         "label": "Polly · Kajal neural  (Indian female, faster)",
     "group": "Cloud — Amazon Polly (Mumbai, needs IAM perm)"},
    {"key": "polly-Aditi-standard",       "label": "Polly · Aditi  (Indian bilingual en-IN/hi-IN)",
     "group": "Cloud — Amazon Polly (Mumbai, needs IAM perm)"},
    {"key": "polly-Ruth-generative",      "label": "Polly · Ruth generative  (US female, premium)",
     "group": "Cloud — Amazon Polly (Mumbai, needs IAM perm)"},
    {"key": "polly-Joanna-generative",    "label": "Polly · Joanna generative  (US female)",
     "group": "Cloud — Amazon Polly (Mumbai, needs IAM perm)"},
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
