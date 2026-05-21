"""
Preset stack combos for the precall config page.

Each combo is a complete STT × LLM × TTS pipeline that can be dialled as
a unit. Combos are curated for the providers actually configured on this
workspace (Deepgram, NVIDIA NIM, AWS Bedrock, Cartesia, ElevenLabs) —
Groq-only and Ollama-only combos were removed because:
  - Groq is blocked by Cloudflare from Indian residential ISPs
  - Ollama isn't running by default on this machine

Pipeline format:
    STT (speech → text)  →  LLM (text → text)  →  TTS (text → speech)

Live first-token latencies (avg from Indian residential ISP):
  NVIDIA NIM US H100         ~388ms  — fastest cloud LLM here
  AWS Bedrock Mumbai (apac.) ~812ms  — geographically closest
  AWS Bedrock us-east-1      ~1100ms — for models not in Mumbai (Mistral)
  Local Ollama (if running)  hardware-dependent

The platform's main agent uses bedrock-mistral.ministral-3-14b-instruct
(us-east-1) so the "Match-main-model" combo below pairs that LLM with
the same STT/TTS used elsewhere, making voice calls behave like the rest
of the product.
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
    quality: str = "average"


# Default = first entry. Picked when no combo/stt/llm/tts is specified.
#
# Each pipeline tells a distinct story (low-latency / India-geo /
# matches-main-model / cheap / premium-voice / best-reasoning / privacy)
# so an operator can A/B by combo without juggling dropdowns.
PRESETS: list[Combo] = [
    # ═══════════════════════════════════════════════════════════════════
    # 🏆 LOW-LATENCY DEFAULT — best of each layer, all configured, all cloud
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="lowlatency-default",
        label="🏆 Low-latency default (Deepgram + NIM + Cartesia)",
        description=(
            "STT: Deepgram Nova-3 (best Indian English)  →  "
            "LLM: NVIDIA NIM Llama 3.3 70B (H100, ~388ms)  →  "
            "TTS: Cartesia Sonic-2 (<100ms streaming). "
            "Strongest end-to-end latency on this workspace. The default."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="default · fastest",
        quality="bestest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🇮🇳 INDIA-ROUTED — Bedrock Mumbai for geo-optimal latency to Indian
    # callers (LLM hop stays in ap-south-1).
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
    # 🤝 MATCH-MAIN-MODEL — uses the same Mistral Ministral 14B that
    # powers the rest of the product, so voice calls share behavior with
    # chat / agents / custom-agent runs.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="match-main-mistral",
        label="🤝 Match main app (Bedrock Mistral 14B)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Mistral Ministral 14B via Bedrock us-east-1 (same model as "
            "the platform's chat / custom-agents — voice replies sound like "
            "the rest of the product)  →  "
            "TTS: Cartesia Sonic-2."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-mistral.ministral-3-14b-instruct",
        tts="cartesia-sonic-2",
        badge="consistent voice",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 💰 CHEAPEST CLOUD — sub-cent per turn, viable production quality.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="budget-cloud",
        label="💰 Cheapest cloud (Nova Micro + Aura)",
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
    # 🧠 SMART B2B — Nemotron is NVIDIA's chat-tuned 70B with the best
    # multi-turn reasoning for high-stakes conversations.
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
    # 🇮🇳 INDIA SWEET SPOT — Mumbai LLM + warm female Indian-English TTS
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="india-sweetspot",
        label="🇮🇳 India sweet spot (Nova Lite + Aura Thalia)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: Amazon Nova Lite via Bedrock ap-south-1 ($0.05/10k turns)  →  "
            "TTS: Deepgram Aura-2 Thalia (warm female voice). "
            "Cheap LLM + decent voice, all geo-routed for Indian callers."
        ),
        stt="deepgram-nova-3",
        llm="bedrock-apac.amazon.nova-lite-v1:0",
        tts="deepgram-aura-2-thalia-en",
        badge="balanced India",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 💎 VOICE-CLONING PREMIUM — your own voice via ElevenLabs.
    # Requires ELEVENLABS_VOICE_ID set in .env (already configured).
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="voice-cloning",
        label="💎 Voice cloning (your voice via ElevenLabs)",
        description=(
            "STT: Deepgram Nova-3  →  "
            "LLM: NVIDIA Llama 3.3 70B  →  "
            "TTS: ElevenLabs Turbo v2.5 with your cloned voice. "
            "Pulls ELEVENLABS_VOICE_ID from .env (already set on this workspace)."
        ),
        stt="deepgram-nova-3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="elevenlabs-eleven_turbo_v2_5",
        badge="paid · cloned voice",
        quality="bestest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🚀 DEEPGRAM + GROQ + DEEPGRAM — best speed-per-rupee
    # Deepgram Nova-3 STT (best Indian English) + Groq Llama 3.3 70B LLM
    # (~250ms LPU silicon) + Deepgram Aura-2 TTS (~$0.015/min, cheapest
    # premium streaming TTS). Sweet spot when you want Groq's speed but
    # don't want ElevenLabs' pricing.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="deepgram-groq-deepgram",
        label="🚀 Deepgram + Groq + Deepgram (cheapest fast combo)",
        description=(
            "STT: Deepgram Nova-3 (best Indian English, ~200ms)  →  "
            "LLM: Groq Llama 3.3 70B Versatile (~250ms LPU)  →  "
            "TTS: Deepgram Aura-2 Thalia (~300ms, cheapest premium streaming TTS). "
            "~600-900ms end-to-end. Best speed-per-rupee combo when ElevenLabs "
            "pricing is too steep but you still want Groq's LLM speed."
        ),
        stt="deepgram-nova-3",
        llm="groq-llama-3.3-70b-versatile",
        tts="deepgram-aura-2-thalia-en",
        badge="fast + cheap",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # ⚡ SPEED ALL-GROQ — Groq's LPU silicon gives the fastest LLM hop
    # of any provider here. Historically blocked by Cloudflare on Indian
    # residential ISPs — user is testing whether their current ISP/key
    # combo gets through. If it does, this is the fastest combo on the
    # whole list.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="speed-allgroq",
        label="⚡ Speed all-Groq (test if your ISP allows it)",
        description=(
            "STT: Groq Whisper Large-v3 (~150ms)  →  "
            "LLM: Groq Llama 3.3 70B Versatile (~250ms LPU silicon)  →  "
            "TTS: ElevenLabs Flash v2.5 (fastest premium TTS). "
            "If Groq isn't blocked on your network, this is the fastest combo here. "
            "If it fails with a connection error, your ISP is on Cloudflare's blocklist — "
            "use one of the other combos."
        ),
        stt="groq-whisper-large-v3",
        llm="groq-llama-3.3-70b-versatile",
        tts="elevenlabs-eleven_flash_v2_5",
        badge="fastest if it works",
        quality="bestest",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🎤 GROQ-STT MIX — keep Groq for STT only (fastest STT we have)
    # but route LLM through NIM. Some ISPs let Groq's STT through but
    # block their chat API; this combo tests that.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="groq-stt-mix",
        label="🎤 Groq STT + NIM LLM + Cartesia",
        description=(
            "STT: Groq Whisper Large-v3 (~150ms LPU)  →  "
            "LLM: NVIDIA NIM Llama 3.3 70B (~388ms)  →  "
            "TTS: Cartesia Sonic-2. "
            "Picks Groq's fast STT while keeping the LLM on a provider that always works."
        ),
        stt="groq-whisper-large-v3",
        llm="nvidia-meta/llama-3.3-70b-instruct",
        tts="cartesia-sonic-2",
        badge="if Groq works",
        quality="best",
    ),

    # ═══════════════════════════════════════════════════════════════════
    # 🔒 PRIVACY HYBRID — audio reasoning never leaves the box.
    # Needs `ollama serve` running locally. Skipped if Ollama is offline.
    # ═══════════════════════════════════════════════════════════════════
    Combo(
        key="privacy-hybrid",
        label="🔒 Privacy hybrid (local STT + LLM, cloud TTS)",
        description=(
            "STT: faster-whisper base on CPU  →  "
            "LLM: Llama 3.1 8B on local Ollama  →  "
            "TTS: Cartesia Sonic-2 (only the voice synthesis is cloud). "
            "Sensitive prompts never reach a cloud LLM. "
            "Requires `ollama serve` to be running."
        ),
        stt="local-whisper-base",
        llm="ollama-llama3.1:8b-instruct-q4_K_M",
        tts="cartesia-sonic-2",
        badge="needs ollama",
        quality="average",
    ),
]


# ─── Manual override dropdowns on the precall page ────────────────────
# Trimmed to providers actually configured on this workspace. Groq removed
# (not configured + EC2-only). Local Whisper kept for the privacy combo.
STT_OPTIONS = [
    {"key": "deepgram-nova-3",        "label": "Deepgram · Nova-3  (best Indian English) ★",
     "group": "Cloud — Deepgram"},
    {"key": "deepgram-nova-2",        "label": "Deepgram · Nova-2  (older, cheaper)",
     "group": "Cloud — Deepgram"},
    {"key": "groq-whisper-large-v3",  "label": "Groq · Whisper Large-v3  (~150ms · test if your ISP allows)",
     "group": "Cloud — Groq (test if reachable)"},
    {"key": "local-whisper-tiny",     "label": "Local · Whisper tiny  (CPU, offline)",
     "group": "Local — Whisper"},
    {"key": "local-whisper-base",     "label": "Local · Whisper base  (slower, more accurate)",
     "group": "Local — Whisper"},
]

LLM_OPTIONS = [
    # NVIDIA NIM — fastest cloud option from Indian residential
    {"key": "nvidia-meta/llama-3.3-70b-instruct",
        "label": "Llama 3.3 70B  · ~388ms ★ default",
        "group": "🚀 NVIDIA NIM (US H100)"},
    {"key": "nvidia-nvidia/llama-3.1-nemotron-70b-instruct",
        "label": "Nemotron 70B (NVIDIA-tuned chat)",
        "group": "🚀 NVIDIA NIM (US H100)"},
    {"key": "nvidia-meta/llama-3.1-8b-instruct",
        "label": "Llama 3.1 8B  (smaller, faster)",
        "group": "🚀 NVIDIA NIM (US H100)"},
    {"key": "nvidia-mistralai/mistral-7b-instruct-v0.3",
        "label": "Mistral 7B v0.3",
        "group": "🚀 NVIDIA NIM (US H100)"},
    # Bedrock Mumbai — geo-optimal for Indian callers
    {"key": "bedrock-apac.amazon.nova-lite-v1:0",
        "label": "Nova Lite  · ~812ms · $0.05/10k",
        "group": "🇮🇳 Bedrock Mumbai (ap-south-1)"},
    {"key": "bedrock-apac.amazon.nova-pro-v1:0",
        "label": "Nova Pro  · ~848ms · $0.76/10k",
        "group": "🇮🇳 Bedrock Mumbai (ap-south-1)"},
    {"key": "bedrock-apac.amazon.nova-micro-v1:0",
        "label": "Nova Micro  · ~848ms · $0.03/10k cheapest",
        "group": "🇮🇳 Bedrock Mumbai (ap-south-1)"},
    # Bedrock US — for models not (yet) in Mumbai
    {"key": "bedrock-mistral.ministral-3-14b-instruct",
        "label": "Mistral Ministral 14B  · same model as the chat app",
        "group": "🌐 Bedrock US (us-east-1)"},
    # Groq — fastest LPU silicon. Blocked on some Indian residential ISPs.
    {"key": "groq-llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B Versatile  · ~250ms (test if your ISP allows)",
        "group": "⚡ Groq (test if reachable)"},
    {"key": "groq-llama-3.1-8b-instant",
        "label": "Llama 3.1 8B Instant  · ~150ms (test if your ISP allows)",
        "group": "⚡ Groq (test if reachable)"},
    # Local Ollama — only useful if `ollama serve` is running
    {"key": "ollama-llama3.1:8b-instruct-q4_K_M",
        "label": "Llama 3.1 8B  (needs Ollama running)",
        "group": "📦 Local Ollama (needs `ollama serve`)"},
    {"key": "ollama-llama3.2:3b",
        "label": "Llama 3.2 3B  (faster local)",
        "group": "📦 Local Ollama (needs `ollama serve`)"},
]

TTS_OPTIONS = [
    # Cartesia — emotionally expressive, our default
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
    {"key": "elevenlabs-eleven_turbo_v2_5",   "label": "Turbo v2.5  (paid, expressive)",
     "group": "Cloud — ElevenLabs"},
    {"key": "elevenlabs-eleven_flash_v2_5",   "label": "Flash v2.5  (paid, fastest premium)",
     "group": "Cloud — ElevenLabs"},
]


def find_combo(key: str) -> Combo | None:
    for c in PRESETS:
        if c.key == key:
            return c
    return None


def default_combo() -> Combo:
    return PRESETS[0]
