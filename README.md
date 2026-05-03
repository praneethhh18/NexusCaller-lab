# NexusCaller Lab — 3-Layer Pipeline Playground

A Pipecat-based sandbox where you swap STT / LLM / TTS from dropdowns and
see how the combined pipeline feels end-to-end. Browser-based, no phone yet —
that's Phase 1 of the main NexusCaller build.

**Goal:** pick the best open-source CPU-friendly stack *before* you wire up
SIP and commit to a pipeline.

---

## Supported models out of the box

### STT (20 options)
| Family | Count | License | Notes |
|---|---|---|---|
| faster-whisper | 6 (tiny → large-v3-turbo) | MIT | The default set everyone should try |
| Distil-Whisper | 3 (small-en, medium-en, large-v3) | MIT | 5–6× faster, English-only |
| Moonshine | 2 (tiny, base) | MIT | Low-latency, CPU-first |
| Parakeet-TDT | 1 | CC-BY-4.0 | GPU recommended |

### LLM (23 options via Ollama)
Grouped in the dropdown by size tier so you pick by CPU budget, not by name:
- **Tiny (<1B)** — smollm2 135M/360M, qwen2.5 0.5B, tinyllama 1.1B
- **Small (1–2B)** — smollm2 1.7B, qwen2.5 1.5B ★, llama3.2 1B, deepseek-r1 1.5B
- **Medium (2–4B)** — gemma2 2B, granite3 2B, qwen2.5 3B, llama3.2 3B, phi3.5 3.8B, nemotron-mini 4B
- **Large (7–8B)** — qwen2.5 7B, llama3.1 8B, mistral 7B, hermes3 8B, qwen2.5-coder 7B, gemma2 9B
- **GPU Only** — qwen2.5 14B, mistral-small 24B, deepseek-r1 7B

### TTS (35 options)
- **Kokoro-82M** (Apache 2.0) — 10 voices (`af_*`, `am_*`, `bf_*`, `bm_*`)
- **Piper** (MIT) — 7 English voices (Amy, Ryan, Kathleen, Lessac, Joe, Jenny, Alba)
- **MeloTTS** (MIT) — first-class EN-US shortcut for the Moonshine + SmolLM2 test combo
- **Coqui XTTS v2** — voice cloning, license caveat (CPML non-commercial)
- **Custom wrappers** — stubs for F5-TTS, VoxCPM, Qwen3-TTS, IndexTTS 2, StyleTTS 2, Bark, Parler-TTS, ChatTTS, MeloTTS, OmniVoice. See `pipeline/tts_custom.py` for the adapter pattern.

Every entry in every dropdown shows its **licence badge**:
- 🟢 green (Apache 2.0 / MIT / BSD) — commercial-safe
- 🟡 yellow (Llama, Gemma, CC-BY) — check the specific license terms
- 🔴 red (CC-BY-NC / CPML / research-only) — don't ship

---

## Quick start (Windows)

```bat
python -m venv venv
venv\Scripts\activate

:: Core dependencies (enough to try Whisper + Ollama + Kokoro + Piper)
pip install -r requirements.txt

:: Make sure Ollama is running and pull at least one small model
ollama pull smollm2:1.7b

:: Start the lab
python server.py
```

Open http://localhost:7860 → pick models → click **Start** → speak.

First run of each model pays a one-time download cost. Subsequent runs are instant.

The default stack is now Moonshine tiny + SmolLM2 1.7B + MeloTTS EN-US.
Before selecting MeloTTS, install its optional package:

```bat
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

### Adding extra models

Read the header of `requirements.txt` — each optional extra is independent.
Uncomment the ones whose models you want to test, `pip install -r requirements.txt`
again, and they light up in the dropdown.

For models not in Pipecat's built-in services (F5-TTS, VoxCPM, Qwen3-TTS, etc.),
open `pipeline/tts_custom.py` — it has a ready-made adapter skeleton. Replace
the `_not_implemented(...)` call for that model with a real `TTSService`
subclass. The file's docstring has the full example.

---

## How the playground works

```
/api/models         ← JSON catalog, powers the dropdowns
/ws/audio?stt=&llm=&tts=   ← Pipecat pipeline (binary audio both ways)
/ws/metrics         ← out-of-band JSON events (latency + transcripts)
```

On each turn the latency observer emits:
```json
{
  "type": "turn_metrics",
  "transcript": "what's my pipeline this week",
  "timings_ms": {
    "stt_final": 180,
    "llm_first_token": 230,
    "tts_first_audio": 190,
    "bot_stop": 1450
  }
}
```

The right-hand UI panel shows the most recent turn + history of the last 8.
Swap models without restarting the server — just Stop, change the dropdowns,
Start again.

---

## Hardware assumptions

- **CPU-only target** — no GPU required.
- Tuned for modern desktop CPUs (AVX2+, 8+ cores ideal).
- A 1.5B LLM via Ollama keeps first-token latency under 500ms on 8-core CPUs.
- faster-whisper-small int8 + Kokoro runs comfortably on 4 cores.
- Large LLMs (7B+) and large Whisper variants are included for **quality
  benchmarking**, not real-time use on CPU.

---

## License discipline

Each registry entry carries a `license` field. The UI badge tells you
immediately whether a model is safe to ship. That said: **always read the
model's LICENSE file on Hugging Face before committing**. Licenses change
(XTTS flipped to paid commercial, F5-TTS added an NC clause, Llama keeps
refining terms).

The commercial-safe intersection as of right now:
- **STT:** any Whisper/Distil-Whisper/Moonshine variant
- **LLM:** Apache 2.0 tier — qwen2.5 family, mistral, smollm2, granite3, phi3.5
- **TTS:** Kokoro (fixed voices), Piper (fixed voices), StyleTTS 2 (fixed voices), MeloTTS, Parler-TTS, OmniVoice

For voice cloning + commercial use you're currently left with a small set,
and most require a paid commercial license. Factor that into product design.

---

## What's NOT in here

- **No SIP / telephony** — that's Phase 1 of the main NexusCaller build.
- **No voice cloning by default** — Kokoro/Piper both use fixed voices, which
  keeps the lab commercial-safe. XTTS + F5 stubs exist for experimentation.
- **No multi-tenant / auth** — single-user local sandbox only.
- **No persistence** — each browser reload clears metric history. Add JSONL
  logging in `pipeline/latency.py` if you want longer-term tracking.

---

## Project layout

```
nexuscaller-lab/
├── server.py                 ← FastAPI + WebSockets + Pipecat runner
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── registry.py           ← THE catalog — add models here
│   ├── builder.py            ← assembles a Pipecat pipeline from 3 keys
│   ├── latency.py            ← per-turn TTFT/TTFA/e2e observer
│   └── tts_custom.py         ← adapter hub for non-Pipecat TTS models
└── web/
    └── index.html            ← single-page UI
```
