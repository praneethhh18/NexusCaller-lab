"""
Preset stack combos for the precall config page.

Each combo is a complete voice-agent pipeline:
    STT (speech → text) → LLM (text → text) → TTS (text → speech)

The descriptions explicitly call out each stage's provider, model, and
its measured first-token / first-byte latency from this network so you
can pick the right trade-off for any given call.

Live numbers come from voice_agent/bench_llms.py and bench_now.py runs
against this AWS account from an Indian residential ISP.
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
    badge: str = ""   # optional pill ("default", "fastest", ...)


# Default = first entry. Picked when no combo/stt/llm/tts is specified.
PRESETS: list[Combo] = [
    # ─── 🚀 FASTEST (live-benchmarked from India) ─────────────────────
    Combo(
        key="fast-nvidia-70b",
        label="🚀 NVIDIA NIM · Llama 3.3 70B",
        description=(
            "STT: Deepgram Nova-3 (~150ms) → "
            "LLM: NVIDIA NIM Llama 3.3 70B on H100 (~388ms) → "
            "TTS: Deepgram Aura-2 Asteria (~150ms). "
            "Best end-to-end latency from India."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="default · fastest",
    ),
    Combo(
        key="fast-nvidia-nemotron",
        label="🚀 NVIDIA NIM · Nemotron 70B (NVIDIA-tuned)",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: NVIDIA Llama 3.1 Nemotron 70B (NVIDIA-tuned for chat) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Slightly higher quality than vanilla Llama, similar latency."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="NVIDIA-tuned",
    ),

    # ─── 🇮🇳 MUMBAI BEDROCK — geographically closest path ──────────────
    Combo(
        key="mumbai-nova-lite",
        label="🇮🇳 Bedrock Mumbai · Nova Lite",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 Mumbai (~812ms) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Cheapest cloud LLM, hosted closest to Indian ISPs ($0.05/10k turns)."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest",
    ),
    Combo(
        key="mumbai-nova-pro",
        label="🇮🇳 Bedrock Mumbai · Nova Pro",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Amazon Nova Pro via Bedrock ap-south-1 Mumbai (~848ms) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Better reasoning than Lite, still Mumbai-routed."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-pro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="balanced",
    ),
    Combo(
        key="mumbai-nova-micro",
        label="🇮🇳 Bedrock Mumbai · Nova Micro (ultra-cheap)",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Amazon Nova Micro via Bedrock ap-south-1 Mumbai (~848ms) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Tiny model, $0.03/10k turns — cheapest of all."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-micro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="ultra-cheap",
    ),

    # ─── 🇺🇸 US BEDROCK — fallback when Mumbai or NIM unavailable ──────
    Combo(
        key="us-llama-70b",
        label="🇺🇸 Bedrock US · Llama 3.3 70B",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Meta Llama 3.3 70B via Bedrock us-east-1 (variable: 530-1500ms) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Open-source Meta 70B; latency varies with India→US network."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-3-70b-instruct-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="open-source",
    ),
    Combo(
        key="us-llama-8b",
        label="🇺🇸 Bedrock US · Llama 3.1 8B",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Meta Llama 3.1 8B via Bedrock us-east-1 (variable) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Cheap open-source 8B."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-1-8b-instruct-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheap open",
    ),

    # ─── 🧠 QUALITY (Anthropic) ────────────────────────────────────────
    Combo(
        key="quality-claude-haiku",
        label="🧠 Bedrock US · Claude Haiku 4.5",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Anthropic Claude Haiku 4.5 via Bedrock us-east-1 (~1240ms) → "
            "TTS: Deepgram Aura-2 Orion (male). "
            "Smart but slow; costs ~$1.22/10k turns."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tts="deepgram-aura-2-orion-en",
        badge="quality",
    ),
    Combo(
        key="quality-claude-sonnet",
        label="🧠 Bedrock US · Claude Sonnet 4.5",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Anthropic Claude Sonnet 4.5 via Bedrock us-east-1 (~2535ms) → "
            "TTS: Deepgram Aura-2 Orion (male). "
            "Best reasoning; use only for high-stakes calls (~$4.14/10k turns)."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        tts="deepgram-aura-2-orion-en",
        badge="premium",
    ),

    # ─── 🔁 ALTERNATIVES ───────────────────────────────────────────────
    Combo(
        key="alt-gemini-flash",
        label="🔁 Google · Gemini 2.5 Flash (free)",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Google Gemini 2.5 Flash via AI Studio (free quota, ~1259ms) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Free Google API quota; backup when AWS keys are exhausted."
        ),
        stt="deepgram-nova-3",
        llm="gemini-gemini-2.5-flash",
        tts="deepgram-aura-2-asteria-en",
        badge="free LLM",
    ),
    Combo(
        key="alt-ollama-local",
        label="🔁 Local · Ollama Llama 3.1 8B",
        description=(
            "STT: Deepgram Nova-3 → "
            "LLM: Llama 3.1 8B running locally on Ollama (no network) → "
            "TTS: Deepgram Aura-2 Asteria. "
            "Zero LLM cost; latency depends on local CPU/GPU."
        ),
        stt="deepgram-nova-3",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="deepgram-aura-2-asteria-en",
        badge="local LLM",
    ),

    # ─── 📦 OFFLINE — fully local, no cloud at all ────────────────────
    Combo(
        key="offline-local",
        label="📦 Offline · all-local stack",
        description=(
            "STT: faster-whisper tiny on CPU → "
            "LLM: Qwen2.5 0.5B on Ollama → "
            "TTS: Piper en_US-lessac on CPU. "
            "Zero cloud calls, real-time on a modern CPU."
        ),
        stt="local-whisper-tiny",
        llm="ollama-qwen2.5:0.5b-instruct",
        tts="piper-en_US-lessac-medium",
        badge="offline",
    ),
]


# ─── Catalog for manual STT/LLM/TTS overrides in the precall page ─────
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
    # NVIDIA NIM — fastest available from India
    {"key": "nvidia-meta/llama-3.3-70b-instruct",
                                          "label": "NVIDIA NIM · Llama 3.3 70B  · 388ms ★ default",
     "group": "Cloud — NVIDIA NIM (US H100)"},
    {"key": "nvidia-meta/llama-3.1-8b-instruct",
                                          "label": "NVIDIA NIM · Llama 3.1 8B",
     "group": "Cloud — NVIDIA NIM (US H100)"},
    {"key": "nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
                                          "label": "NVIDIA NIM · Nemotron 70B  (tuned)",
     "group": "Cloud — NVIDIA NIM (US H100)"},
    {"key": "nvidia-mistralai/mistral-7b-instruct-v0.3",
                                          "label": "NVIDIA NIM · Mistral 7B v0.3",
     "group": "Cloud — NVIDIA NIM (US H100)"},
    {"key": "nvidia-deepseek-ai/deepseek-r1",
                                          "label": "NVIDIA NIM · DeepSeek R1  (reasoning)",
     "group": "Cloud — NVIDIA NIM (US H100)"},
    # Bedrock Mumbai — closest geographic path
    {"key": "bedrock-apac.amazon.nova-lite-v1:0",
                                          "label": "Bedrock Mumbai · Nova Lite  · 812ms · $0.05/10k",
     "group": "Cloud — AWS Bedrock (ap-south-1 Mumbai)"},
    {"key": "bedrock-apac.amazon.nova-pro-v1:0",
                                          "label": "Bedrock Mumbai · Nova Pro  · 848ms · $0.76/10k",
     "group": "Cloud — AWS Bedrock (ap-south-1 Mumbai)"},
    {"key": "bedrock-apac.amazon.nova-micro-v1:0",
                                          "label": "Bedrock Mumbai · Nova Micro  · 848ms · $0.03/10k",
     "group": "Cloud — AWS Bedrock (ap-south-1 Mumbai)"},
    # Bedrock US — fallback
    {"key": "bedrock-us.meta.llama3-3-70b-instruct-v1:0",
                                          "label": "Bedrock US · Llama 3.3 70B",
     "group": "Cloud — AWS Bedrock (us-east-1)"},
    {"key": "bedrock-us.meta.llama3-1-8b-instruct-v1:0",
                                          "label": "Bedrock US · Llama 3.1 8B",
     "group": "Cloud — AWS Bedrock (us-east-1)"},
    {"key": "bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                          "label": "Bedrock US · Claude Haiku 4.5  (slow + premium)",
     "group": "Cloud — AWS Bedrock (us-east-1)"},
    {"key": "bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                                          "label": "Bedrock US · Claude Sonnet 4.5  (slow + premium)",
     "group": "Cloud — AWS Bedrock (us-east-1)"},
    # Gemini — free fallback
    {"key": "gemini-gemini-2.5-flash",    "label": "Google · Gemini 2.5 Flash  (free, ~1259ms)",
     "group": "Cloud — Google AI"},
    {"key": "gemini-gemini-2.5-flash-lite", "label": "Google · Gemini 2.5 Flash Lite",
     "group": "Cloud — Google AI"},
    {"key": "gemini-gemini-2.5-pro",      "label": "Google · Gemini 2.5 Pro",
     "group": "Cloud — Google AI"},
    # OpenAI / Groq
    {"key": "openai-gpt-4o-mini",         "label": "OpenAI · GPT-4o mini",
     "group": "Cloud — OpenAI"},
    {"key": "openai-gpt-4o",              "label": "OpenAI · GPT-4o",
     "group": "Cloud — OpenAI"},
    {"key": "groq-llama-3.1-8b-instant",  "label": "Groq · Llama 3.1 8B  (CDN-blocked from your ISP)",
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
    {"key": "elevenlabs-eleven_turbo_v2_5",   "label": "ElevenLabs · Turbo v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",   "label": "ElevenLabs · Flash v2.5  (paid plan)",
     "group": "Cloud — ElevenLabs"},
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
