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
    # ─── ⚡ FASTEST + BEST VALUE (live-benchmarked) ──────────────────────
    # Numbers from voice_agent/bench_llms.py against this AWS account.
    Combo(
        key="fast-llama-70b",
        label="⚡ Llama 3.3 70B  (fastest)",
        description="Nova-3 STT · Meta Llama 3.3 70B (Bedrock) · Aura-2. 535ms first token, $0.60/10k turns. Surprisingly the fastest Bedrock model in our benchmark.",
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-3-70b-instruct-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="default",
    ),
    Combo(
        key="fast-llama-8b",
        label="⚡ Llama 3.1 8B  (cheap+fast)",
        description="Nova-3 STT · Meta Llama 3.1 8B (Bedrock) · Aura-2. 656ms first token, $0.17/10k turns. Best cost-per-turn.",
        stt="deepgram-nova-3",
        llm="bedrock-us.meta.llama3-1-8b-instruct-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest fast",
    ),
    Combo(
        key="fast-nova-lite",
        label="⚡ Nova Lite  (cheapest fast)",
        description="Nova-3 STT · Amazon Nova Lite (Bedrock) · Aura-2. 876ms first token, $0.05/10k turns. Cheapest cloud LLM that holds a conversation.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest",
    ),
    Combo(
        key="fast-nova-pro",
        label="⚡ Nova Pro  (Amazon mid-tier)",
        description="Nova-3 STT · Amazon Nova Pro (Bedrock) · Aura-2. 726ms first token, $0.76/10k turns. Better reasoning than Lite.",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-pro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="",
    ),

    # ─── 💰 ULTRA-CHEAP — when budget > polish ──────────────────────────
    Combo(
        key="cheap-nova-micro",
        label="💰 Nova Micro  (ultra cheap)",
        description="Nova-3 STT · Amazon Nova Micro (Bedrock) · Aura-2. $0.03/10k turns — 25× cheaper than Haiku. First-token has cold-start penalty (~2s on first call, faster after warmup).",
        stt="deepgram-nova-3",
        llm="bedrock-us.amazon.nova-micro-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="cheapest of all",
    ),
    # ─── 🚀 NVIDIA NIM — H100-hosted, often beats Bedrock ──────────────
    Combo(
        key="nvidia-llama-70b",
        label="🚀 NVIDIA · Llama 3.3 70B (H100)",
        description="Nova-3 STT · Meta Llama 3.3 70B on NVIDIA NIM · Aura-2. H100 GPUs, free $1000 credits. Should beat Bedrock 70B latency. Sign up at build.nvidia.com.",
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="NIM",
    ),
    Combo(
        key="nvidia-llama-8b",
        label="🚀 NVIDIA · Llama 3.1 8B (H100)",
        description="Nova-3 STT · Meta Llama 3.1 8B on NVIDIA NIM · Aura-2. H100 GPUs — fastest 8B option from India.",
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.1-8b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="NIM fast",
    ),
    Combo(
        key="nvidia-nemotron-70b",
        label="🚀 NVIDIA · Nemotron 70B (H100)",
        description="Nova-3 STT · NVIDIA's tuned Llama 3.1 Nemotron 70B · Aura-2. NVIDIA's own optimized model, often higher quality than vanilla Llama.",
        stt="deepgram-nova-3",
        llm="nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        tts="deepgram-aura-2-asteria-en",
        badge="NVIDIA tuned",
    ),

    Combo(
        key="cheap-jamba-mini",
        label="💰 AI21 Jamba 1.5 Mini  (Mamba-Transformer)",
        description="Nova-3 STT · AI21 Jamba 1.5 Mini (Bedrock) · Aura-2. 545ms first token, $0.20/1M in. Hybrid Mamba+Transformer arch — surprisingly fast.",
        stt="deepgram-nova-3",
        llm="bedrock-ai21.jamba-1-5-mini-v1:0",
        tts="deepgram-aura-2-asteria-en",
        badge="hybrid arch",
    ),

    # ─── 🧠 QUALITY — when call outcome > 100ms latency ─────────────────
    Combo(
        key="quality-claude-haiku",
        label="🧠 Claude Haiku 4.5",
        description="Nova-3 STT · Claude Haiku 4.5 (Bedrock) · Aura-2 Orion. 1240ms first token, $1.22/10k turns. Smartest in the 'fast' tier but ~$1 more per 10k turns.",
        stt="deepgram-nova-3",
        llm="bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
        tts="deepgram-aura-2-orion-en",
        badge="",
    ),
    Combo(
        key="quality-claude-sonnet",
        label="🧠 Claude Sonnet 4.5  (slow but smart)",
        description="Nova-3 STT · Claude Sonnet 4.5 (Bedrock) · Aura-2 Orion. 2.5s first-token, $4.14/10k turns. Use only when outcome quality matters more than feel.",
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
    # NVIDIA NIM — H100-hosted, OpenAI-compat, free $1000 credits
    {"key": "nvidia-meta/llama-3.3-70b-instruct",
                                          "label": "NVIDIA NIM · Llama 3.3 70B  (H100, fastest from India)",
     "group": "Cloud — NVIDIA NIM"},
    {"key": "nvidia-meta/llama-3.1-8b-instruct",
                                          "label": "NVIDIA NIM · Llama 3.1 8B  (H100)",
     "group": "Cloud — NVIDIA NIM"},
    {"key": "nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
                                          "label": "NVIDIA NIM · Nemotron 70B  (NVIDIA-tuned Llama)",
     "group": "Cloud — NVIDIA NIM"},
    {"key": "nvidia-mistralai/mistral-7b-instruct-v0.3",
                                          "label": "NVIDIA NIM · Mistral 7B v0.3",
     "group": "Cloud — NVIDIA NIM"},
    {"key": "nvidia-deepseek-ai/deepseek-r1",
                                          "label": "NVIDIA NIM · DeepSeek R1  (reasoning)",
     "group": "Cloud — NVIDIA NIM"},
    {"key": "nvidia-google/gemma-2-9b-it",
                                          "label": "NVIDIA NIM · Gemma 2 9B",
     "group": "Cloud — NVIDIA NIM"},
    # Bedrock — ranked by live benchmark (first-token latency)
    {"key": "bedrock-us.meta.llama3-1-8b-instruct-v1:0",
                                          "label": "Bedrock · Llama 3.1 8B  ★ default · 656ms · $0.17/10k",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.meta.llama3-3-70b-instruct-v1:0",
                                          "label": "Bedrock · Llama 3.3 70B  · 535ms · $0.60/10k",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.amazon.nova-pro-v1:0",
                                          "label": "Bedrock · Nova Pro  · 726ms · $0.76/10k",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.amazon.nova-lite-v1:0",
                                          "label": "Bedrock · Nova Lite  · 876ms · $0.05/10k (cheapest fast)",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.amazon.nova-micro-v1:0",
                                          "label": "Bedrock · Nova Micro  · cold-start ~2s · $0.03/10k (cheapest)",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-ai21.jamba-1-5-mini-v1:0",
                                          "label": "Bedrock · AI21 Jamba 1.5 Mini  · 545ms · ~$0.36/10k",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-ai21.jamba-1-5-large-v1:0",
                                          "label": "Bedrock · AI21 Jamba 1.5 Large  · 648ms",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.anthropic.claude-haiku-4-5-20251001-v1:0",
                                          "label": "Bedrock · Claude Haiku 4.5  · 1.2s · $1.22/10k (slow for voice)",
     "group": "Cloud — AWS Bedrock"},
    {"key": "bedrock-us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                                          "label": "Bedrock · Claude Sonnet 4.5  · 2.5s · $4.14/10k (quality)",
     "group": "Cloud — AWS Bedrock"},
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
